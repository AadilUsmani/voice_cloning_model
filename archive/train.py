import os
import logging
import torch
from torch.utils.data import Dataset, DataLoader
import modal

logger = logging.getLogger(__name__)

# --- Modal Setup ---
app = modal.App("fyp-training")
volume = modal.Volume.from_name("libritts-volume")

image = modal.Image.debian_slim().pip_install(
    "torch", "numpy", "tqdm", "pytorch-metric-learning"
).add_local_file("model.py", remote_path="/root/model.py")

# --- Hyperparameters ---
BATCH_SIZE = 64
EPOCHS = 10
LEARNING_RATE = 1e-4
M_PER_CLASS = 4
GRAD_CLIP = 5.0
SAVE_DIR = "/data/checkpoints"
NUM_SHARDS = 10


class ShardedMelDataset(Dataset):
    """Loads pre-built .pt shards into RAM. float16 → float32 on load."""

    def __init__(self):
        from tqdm import tqdm

        self.data_cache: list[torch.Tensor] = []
        self.label_cache: list[int] = []

        logger.info(f"Loading {NUM_SHARDS} shards into RAM...")

        for i in tqdm(range(NUM_SHARDS), desc="Loading Shards"):
            path = f"/data/dataset_shard_{i}.pt"
            if not os.path.exists(path):
                logger.warning(f"Missing shard, skipping: {path}")
                continue

            # weights_only=True avoids the PyTorch 2.x deprecation warning
            # and is safer against arbitrary code execution in pickle
            shard = torch.load(path, weights_only=True)
            for tensor, label in shard["data"]:
                self.data_cache.append(tensor.float())  # float16 → float32
                self.label_cache.append(label)

        if not self.data_cache:
            raise RuntimeError("No shard data loaded. Check /data for .pt files.")

        logger.info(f"Loaded {len(self.data_cache)} total chunks.")

    def __len__(self) -> int:
        return len(self.data_cache)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        return self.data_cache[idx], self.label_cache[idx]


@app.function(image=image, volumes={"/data": volume}, timeout=14400, gpu="T4")
def train_model():
    from tqdm import tqdm
    from pytorch_metric_learning import losses, miners
    from pytorch_metric_learning.samplers import MPerClassSampler
    import torch.optim as optim
    from model import SpeakerEncoder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True
    os.makedirs(SAVE_DIR, exist_ok=True)

    # --- Data ---
    dataset = ShardedMelDataset()
    
    # Load speaker mapping from shard 0 to get actual speaker count
    shard_0_path = "/data/dataset_shard_0.pt"
    shard_0 = torch.load(shard_0_path, weights_only=True)
    n_speakers = len(shard_0.get("mapping", {}))
    logger.info(f"Detected {n_speakers} unique speakers in dataset")
    
    # Build the smart sampler
    my_sampler = MPerClassSampler(
        dataset.label_cache, 
        m=M_PER_CLASS, 
        length_before_new_iter=len(dataset)
    )
    
    # Plug it into the DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        sampler=my_sampler,     # <-- This replaces shuffle=True
        drop_last=True,
        num_workers=0,
        pin_memory=True,
    )

    # --- Model ---
    model = SpeakerEncoder(n_speakers=n_speakers).to(device)

    # torch.compile traces the model into optimized kernels — free ~10-20% GPU speedup
    # on PyTorch 2.x with no code changes to the model itself
    model = torch.compile(model)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    miner = miners.BatchHardMiner()
    triplet_loss_func = losses.TripletMarginLoss(margin=0.2)
    ce_loss_func = torch.nn.CrossEntropyLoss().to(device)

    logger.info(f"Starting training on {device} for {EPOCHS} epochs...")
    best_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        total_triplet = 0.0
        total_ce = 0.0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS}")

        for batch_idx, (mels, labels) in enumerate(pbar):
            mels = mels.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            embeddings, logits = model(mels)
            hard_pairs = miner(embeddings, labels)
            loss_triplet = triplet_loss_func(embeddings, labels, hard_pairs)
            loss_ce = ce_loss_func(logits, labels)
            
            # Combine both losses
            loss = loss_triplet + loss_ce

            # Skip backward when no valid triplets exist in the batch
            if loss.item() > 0:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()

            total_loss += loss.item()
            total_triplet += loss_triplet.item()
            total_ce += loss_ce.item()
            pbar.set_postfix({
                "loss": f"{total_loss / (batch_idx + 1):.4f}",
                "triplet": f"{total_triplet / (batch_idx + 1):.4f}",
                "ce": f"{total_ce / (batch_idx + 1):.4f}"
            })

        scheduler.step()
        epoch_loss = total_loss / len(dataloader)
        current_lr = scheduler.get_last_lr()[0]

        # Always save the per-epoch checkpoint into the volume
        ckpt_path = os.path.join(SAVE_DIR, f"encoder_epoch_{epoch + 1}.pth")
        torch.save(model.state_dict(), ckpt_path)

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_path = os.path.join(SAVE_DIR, "encoder_best.pth")
            torch.save(model.state_dict(), best_path)
            # Commit immediately on a new best — most important checkpoint to preserve
            volume.commit()
            logger.info(f"New best saved (loss: {best_loss:.4f}) — volume committed.")

        logger.info(
            f"Epoch {epoch + 1}/{EPOCHS} | loss: {epoch_loss:.4f} | lr: {current_lr:.6f}"
        )

    # Single final commit flushes all per-epoch checkpoints + final weights
    torch.save(model.state_dict(), os.path.join(SAVE_DIR, "encoder_final.pth"))
    volume.commit()
    logger.info("Phase 3 complete. All weights saved.")


@app.local_entrypoint()
def main():
    train_model.remote()

if __name__ == "__main__":
    main()
    