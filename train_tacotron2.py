import os
# CRITICAL VRAM FIX: Prevents memory fragmentation on the T4 GPU
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import logging
import torch
import torch.nn.functional as F
import modal
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# --- Modal Setup ---
app    = modal.App("fyp-tacotron2-training")
volume = modal.Volume.from_name("libritts-volume")

image = (
    modal.Image.debian_slim()
    .pip_install("torch", "torchaudio", "numpy", "matplotlib", "tqdm")
    .add_local_file("synthesizer_dataset.py", remote_path="/root/synthesizer_dataset.py")
    .add_local_file("tacotron2_model.py",     remote_path="/root/tacotron2_model.py")
)

# --- Paths ---
METADATA_PATH  = "/data/synthesizer_dataset/train.txt"
CHECKPOINT_DIR = "/data/tacotron2_checkpoints"
ATTENTION_DIR  = "/data/tacotron2_attention_plots"
INFERENCE_DIR  = "/data/tacotron2_inference_samples"

# --- Hyperparameters (T4 Optimized) ---
BATCH_SIZE                  = 8      # Reduced from 8 to fit in 16GB VRAM
GRADIENT_ACCUMULATION_STEPS = 8      # 2 * 4 = 8 (Simulates batch size 8 mathematically)
LEARNING_RATE               = 1e-3
EPOCHS                      = 100
MAX_MEL_FRAMES              = 600    # Reduced from 1000 to save VRAM on extreme outliers
GRAD_CLIP_THRESH            = 1.0
TF_DECAY_START              = 10_000 # step at which teacher forcing begins to decay
TF_DECAY_STEPS              = 50_000 # steps to decay from 1.0 → 0.5
TF_MIN                      = 0.5

CHECKPOINT_EVERY  = 1_000
VISUALIZE_EVERY   = 1_000
INFERENCE_EVERY   = 5_000

TEST_SENTENCES = [
    "Hello, this is a test.",
    "The quick brown fox jumps over the lazy dog.",
]


# ---------------------------------------------------------------
# Loss (Using L1 for crisp audio fidelity)
# ---------------------------------------------------------------
def get_mask_from_lengths(lengths: torch.Tensor, max_len: int | None = None) -> torch.Tensor:
    if max_len is None:
        max_len = lengths.max().item()
    ids  = torch.arange(max_len, device=lengths.device)
    return ids.unsqueeze(0) < lengths.unsqueeze(1)      # [B, T]


def tacotron_loss(
    mel_pred:    torch.Tensor,
    mel_postnet: torch.Tensor,
    mel_target:  torch.Tensor,
    stop_pred:   torch.Tensor,
    stop_target: torch.Tensor,
    mel_lengths: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mask      = get_mask_from_lengths(mel_lengths, mel_target.shape[2])  # [B, T]
    mel_mask  = mask.float().unsqueeze(1)                                 # [B, 1, T]
    denom     = mel_mask.sum() * mel_target.shape[1] * 2

    # L1 Loss deployed as requested!
    mel_loss  = (
        F.l1_loss(mel_pred    * mel_mask, mel_target * mel_mask, reduction="sum") +
        F.l1_loss(mel_postnet * mel_mask, mel_target * mel_mask, reduction="sum")
    ) / denom

    pos_weight = torch.tensor(10.0, device=stop_target.device)
    stop_loss  = F.binary_cross_entropy_with_logits(
        stop_pred, stop_target, pos_weight=pos_weight, reduction="none"
    )
    stop_loss  = (stop_loss * mask.float()).sum() / mask.float().sum()

    return mel_loss + stop_loss, mel_loss, stop_loss


# ---------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------
def plot_attention(alignment: "np.ndarray", step: int, path: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(alignment, aspect="auto", origin="lower", interpolation="none")
    fig.colorbar(im, ax=ax)
    ax.set_xlabel("Encoder timestep")
    ax.set_ylabel("Decoder timestep")
    ax.set_title(f"Attention Alignment — Step {step}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------
def save_checkpoint(path: str, model, optimizer, scaler, step: int,
                    epoch: int, loss: float) -> None:
    torch.save({
        "step":               step,
        "epoch":              epoch,
        "model_state_dict":   model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict":  scaler.state_dict() if scaler else None,
        "loss":               loss,
    }, path)


def load_checkpoint(path: str, model, optimizer, scaler, device):
    if not os.path.exists(path):
        return 0, 0, float("inf")
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and ckpt.get("scaler_state_dict"):
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        step  = ckpt.get("step",  0)
        epoch = ckpt.get("epoch", 0)
        loss  = ckpt.get("loss",  float("inf"))
        logger.info(f"Resumed from step {step}, epoch {epoch}, loss {loss:.4f}")
        return step, epoch, loss
    except Exception as e:
        logger.error(f"Checkpoint load failed: {e}. Starting fresh.")
        return 0, 0, float("inf")


def teacher_forcing_ratio(step: int) -> float:
    if step < TF_DECAY_START:
        return 1.0
    return max(TF_MIN, 1.0 - (step - TF_DECAY_START) / TF_DECAY_STEPS)


# ---------------------------------------------------------------
# Training function
# ---------------------------------------------------------------
@app.function(image=image, volumes={"/data": volume}, timeout=14400, gpu="T4")
def train_tacotron2():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s"
    )
    sys.path.insert(0, "/root")

    from tqdm import tqdm
    from synthesizer_dataset import get_dataloader, VOCAB, text_to_sequence
    from tacotron2_model import Tacotron2

    volume.reload()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on {device}")

    for d in (CHECKPOINT_DIR, ATTENTION_DIR, INFERENCE_DIR):
        os.makedirs(d, exist_ok=True)

    # --- Data ---
    # CRITICAL CLOUD FIX: num_workers=0 to prevent network volume crashes
    dataloader = get_dataloader(
        METADATA_PATH, 
        batch_size=BATCH_SIZE, 
        max_mel_frames=MAX_MEL_FRAMES
    )
    logger.info(f"Dataset ready — {len(dataloader)} batches/epoch")

    # --- Model ---
    model = Tacotron2(
        vocab_size=len(VOCAB), n_mels=80, speaker_embedding_dim=256
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-6)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-5
    )
    scaler = torch.amp.GradScaler("cuda") if torch.cuda.is_available() else None

    # --- Resume ---
    latest_ckpt = os.path.join(CHECKPOINT_DIR, "latest.pth")
    best_ckpt   = os.path.join(CHECKPOINT_DIR, "best.pth")
    global_step, start_epoch, best_loss = load_checkpoint(
        latest_ckpt, model, optimizer, scaler, device
    )

    logger.info("=" * 60 + "\n  STARTING TRAINING\n" + "=" * 60)

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        epoch_loss  = 0.0
        pbar        = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        needs_commit = False

        optimizer.zero_grad() # Initialize before the loop for accumulation

        for batch_idx, batch in enumerate(pbar):
            if batch is None:
                continue

            global_step += 1
            tf_ratio = teacher_forcing_ratio(global_step)

            text         = batch["text"].to(device)
            text_lengths = batch["text_lengths"].to(device)
            mel          = batch["mel_targets"].to(device)
            mel_lengths  = batch["mel_lengths"].to(device)
            embeddings   = batch["speaker_embeds"].to(device)
            gate_targets = batch["gate_targets"].to(device)

            with torch.amp.autocast("cuda", enabled=scaler is not None):
                mel_postnet, mel_pred, stop_tokens, alignments = model(
                    text, text_lengths, embeddings, mel, tf_ratio
                )
                loss, mel_loss, stop_loss = tacotron_loss(
                    mel_pred, mel_postnet, mel,
                    stop_tokens, gate_targets, mel_lengths
                )

            # CRITICAL VRAM FIX: Scale loss for gradient accumulation
            loss = loss / GRADIENT_ACCUMULATION_STEPS

            if scaler:
                scaler.scale(loss).backward()
                # Only step the optimizer every N steps
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1 == len(dataloader)):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()
            else:
                loss.backward()
                if (batch_idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or (batch_idx + 1 == len(dataloader)):
                    torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_THRESH)
                    optimizer.step()
                    optimizer.zero_grad()
                    torch.cuda.empty_cache()

            # For logging, multiply back to get the true loss value
            loss_val    = loss.item() * GRADIENT_ACCUMULATION_STEPS
            epoch_loss += loss_val
            pbar.set_postfix(loss=f"{loss_val:.4f}", mel=f"{mel_loss.item():.4f}",
                             stop=f"{stop_loss.item():.4f}", tf=f"{tf_ratio:.2f}")

            # --- Periodic checkpoint ---
            if global_step % CHECKPOINT_EVERY == 0:
                try:
                    save_checkpoint(latest_ckpt, model, optimizer, scaler,
                                    global_step, epoch, loss_val)
                    needs_commit = True
                    logger.info(f"Checkpoint saved at step {global_step}")
                except Exception as e:
                    logger.error(f"Checkpoint save failed: {e}")

            # --- Attention plot ---
            if global_step % VISUALIZE_EVERY == 0:
                try:
                    plot_attention(
                        alignments[0].detach().cpu().numpy(),
                        global_step,
                        os.path.join(ATTENTION_DIR, f"attention_step_{global_step}.png"),
                    )
                    needs_commit = True
                except Exception as e:
                    logger.error(f"Attention plot failed: {e}")

            # --- Inference sanity check ---
            if global_step % INFERENCE_EVERY == 0:
                model.eval()
                with torch.no_grad():
                    try:
                        for i, sent in enumerate(TEST_SENTENCES):
                            seq  = torch.LongTensor(text_to_sequence(sent)).unsqueeze(0).to(device)
                            slen = torch.LongTensor([seq.size(1)]).to(device)
                            emb  = embeddings[0:1]

                            m_post, _, _, align = model(
                                seq, slen, emb, mels=None, teacher_forcing_ratio=0.0
                            )
                            prefix = os.path.join(INFERENCE_DIR, f"step_{global_step}_test_{i}")
                            torch.save(m_post.cpu(), f"{prefix}.pt")
                            plot_attention(align[0].detach().cpu().numpy(),
                                          global_step, f"{prefix}_attention.png")
                        needs_commit = True
                    except Exception as e:
                        logger.error(f"Inference test failed: {e}")
                model.train()

            # Single commit per interval window
            if needs_commit:
                volume.commit()
                needs_commit = False

        # --- End of epoch ---
        avg_loss = epoch_loss / len(dataloader)
        scheduler.step(avg_loss)
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1}/{EPOCHS} complete — "
            f"avg_loss={avg_loss:.4f}  lr={current_lr:.2e}"
        )

        if avg_loss < best_loss:
            best_loss = avg_loss
            try:
                save_checkpoint(best_ckpt, model, optimizer, scaler,
                                global_step, epoch, best_loss)
                volume.commit()
                logger.info(f"New best model saved (loss={best_loss:.4f})")
            except Exception as e:
                logger.error(f"Best checkpoint save failed: {e}")

    logger.info(f"Training complete. Best loss: {best_loss:.4f}")

@app.local_entrypoint()
def main():
    train_tacotron2.remote()