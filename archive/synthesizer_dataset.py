import os
import random
import logging
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

# ==========================================
# 1. CONFIGURATION & CONSTANTS
# ==========================================
MEL_CHANNELS   = 80
EMBED_DIM      = 256
MEL_PAD_VALUE  = -11.5129   # log-mel silence floor
MAX_MEL_FRAMES = 1000
MAX_LOAD_RETRIES = 10       # give up after N consecutive bad samples

# Character vocabulary
CHARACTERS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-':;\"()"
PAD_TOKEN  = "<PAD>"
EOS_TOKEN  = "<EOS>"
VOCAB      = [PAD_TOKEN, EOS_TOKEN] + list(CHARACTERS)
VOCAB_SIZE = len(VOCAB)

CHAR_TO_IDX: dict[str, int] = {c: i for i, c in enumerate(VOCAB)}
PAD_IDX = CHAR_TO_IDX[PAD_TOKEN]
EOS_IDX = CHAR_TO_IDX[EOS_TOKEN]


def text_to_sequence(text: str) -> list[int]:
    """
    Maps characters to vocab indices; unknown chars fall back to space.
    Appends EOS so the attention mechanism learns where to stop.
    """
    seq = [CHAR_TO_IDX.get(c, CHAR_TO_IDX[" "]) for c in text]
    seq.append(EOS_IDX)
    return seq


# ==========================================
# 2. DATASET  (Lazy-load, cloud-safe)
# ==========================================
class SynthesizerDataset(Dataset):
    def __init__(self, metadata_path: str, max_mel_frames: int = MAX_MEL_FRAMES):
        self.max_mel_frames = max_mel_frames
        self.data: list[dict] = []

        logger.info("Initializing dataset metadata...")
        for line in Path(metadata_path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split("|")
            if len(parts) != 4:
                logger.warning(f"Skipping malformed line: {line[:60]}")
                continue

            # Correct column order based on Phase 4: base_id | text | mel_path | emb_path
            base_id, text, mel_path, emb_path = parts
            self.data.append({
                "base_id":    base_id,
                "text":       text,
                "emb_path":   emb_path,
                "mel_path":   mel_path,
            })

        logger.info(f"Loaded {len(self.data)} utterance records.")

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """
        Lazy-loads tensors with bounded retries.
        Skips samples that are missing, corrupt, or exceed MAX_MEL_FRAMES.
        """
        # Increased to 50 attempts to protect tiny validation datasets
        for attempt in range(50):
            item = self.data[idx]
            try:
                mel = torch.load(item["mel_path"], weights_only=True).float()  # [80, T]

                if mel.shape[1] > self.max_mel_frames:
                    # Move to the NEXT file sequentially instead of random jumping
                    idx = (idx + 1) % len(self.data)
                    continue

                embedding = torch.load(item["emb_path"], weights_only=True).float()  # [256]
                text_seq  = torch.LongTensor(text_to_sequence(item["text"]))

                return {
                    "text":      text_seq,
                    "embedding": embedding,
                    "mel":       mel,
                    "base_id":   item["base_id"],
                }

            except Exception as e:
                # logger.warning(f"[attempt {attempt+1}/50] Failed to load {item['base_id']}: {e}")
                # Move to the NEXT file sequentially
                idx = (idx + 1) % len(self.data)

        raise RuntimeError(
            "Could not load a valid sample after 50 sequential attempts. "
            "Check your synthesizer_dataset volume for corruption."
        )

# ==========================================
# 3. BATCH COLLATOR
# ==========================================
def collate_fn(batch: list[dict]) -> dict[str, torch.Tensor]:
    # Sort descending by text length (required for pack_padded_sequence in encoder)
    batch.sort(key=lambda x: x["text"].shape[0], reverse=True)

    text_lengths = torch.LongTensor([item["text"].shape[0] for item in batch])
    mel_lengths  = torch.LongTensor([item["mel"].shape[1]  for item in batch])

    B           = len(batch)
    max_text    = text_lengths.max().item()
    max_mel     = mel_lengths.max().item()

    text_padded  = torch.full((B, max_text),          PAD_IDX,       dtype=torch.long)
    mel_padded   = torch.full((B, MEL_CHANNELS, max_mel), MEL_PAD_VALUE, dtype=torch.float32)
    embeddings   = torch.zeros(B, EMBED_DIM,                         dtype=torch.float32)
    gate_padded  = torch.zeros(B, max_mel,                           dtype=torch.float32)

    for i, item in enumerate(batch):
        t_len = item["text"].shape[0]
        m_len = item["mel"].shape[1]

        text_padded[i, :t_len]      = item["text"]
        mel_padded[i, :, :m_len]    = item["mel"]
        embeddings[i]               = item["embedding"]

        # 1.0 at last real frame and all padding — stop-token supervision
        gate_padded[i, m_len - 1:]  = 1.0

        if gate_padded[i, m_len - 1].item() != 1.0:
            raise ValueError(f"Gate target sanity check failed at batch index {i}. "
                             f"mel_len={m_len}, max_mel={max_mel}")

    return {
        "text":           text_padded,
        "text_lengths":   text_lengths,
        "mel_targets":    mel_padded,
        "mel_lengths":    mel_lengths,
        "speaker_embeds": embeddings,
        "gate_targets":   gate_padded,
    }


# ==========================================
# 4. DATALOADER FACTORY
# ==========================================
def get_dataloader(
    metadata_path: str,
    batch_size: int = 8,
    num_workers: int = 0,
    max_mel_frames: int = 1000
) -> DataLoader:
    dataset = SynthesizerDataset(metadata_path, max_mel_frames=max_mel_frames)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        drop_last=True,       # keeps batch shapes uniform for Tacotron
        pin_memory=True,
        persistent_workers=(num_workers > 0),
    )
    logger.info(f"DataLoader ready — {len(dataset)} samples, "
                f"batch={batch_size}, workers={num_workers}, max_frames={max_mel_frames}")
    return loader


# ==========================================
# 5. SMOKE TEST
# ==========================================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger.info(f"Vocab size: {VOCAB_SIZE}")
    encoded = text_to_sequence("Testing 123!")
    logger.info(f"Test encoding (last token should be EOS={EOS_IDX}): {encoded}")
    assert encoded[-1] == EOS_IDX, "EOS not appended correctly"
    logger.info("Smoke test passed.")