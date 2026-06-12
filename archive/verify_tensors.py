import modal
# REMOVED: import torch (moved inside the function below)
import random
import logging
from pathlib import Path
from collections import defaultdict

logger = logging.getLogger(__name__)

app    = modal.App("fyp-verify-data")
volume = modal.Volume.from_name("libritts-volume")

# 1. Define the custom image
image = modal.Image.debian_slim().pip_install("torch")

META_PATH  = "/data/synthesizer_dataset/train.txt"
SAMPLE_SIZE = 500
MIN_MEL_LEN = 10    # frames — anything shorter is likely a bad file

# 2. ATTACH THE IMAGE HERE (Added image=image)
@app.function(image=image, volumes={"/data": volume})
def verify_dataset():
    # 3. IMPORT TORCH HERE (Inside the function)
    import torch 
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Reloading volume...")
    volume.reload()

    lines = Path(META_PATH).read_text(encoding="utf-8").strip().splitlines()
    logger.info(f"Found {len(lines)} entries in train.txt")

    # --- Structural validation: every line must have 4 pipe-separated fields ---
    malformed = [i for i, ln in enumerate(lines, 1) if len(ln.split("|")) != 4]
    if malformed:
        logger.error(f"Malformed lines (wrong field count): {len(malformed)} — e.g. line {malformed[0]}")
    else:
        logger.info("All lines have correct 4-field structure.")

    # --- Random sample for tensor-level checks ---
    sample = random.sample(lines, min(SAMPLE_SIZE, len(lines)))
    errors = defaultdict(list)   # category -> list of offending paths

    for line in sample:
        parts                          = line.split("|")
        base_id, text, mel_path, emb_path = parts  # Corrected order!
        
        # 1. Text sanity
        if not text.strip():
            errors["empty_text"].append(base_id)
            continue

        # 2. File existence
        if not Path(emb_path).exists():
            errors["missing_file"].append(emb_path)
            continue
        if not Path(mel_path).exists():
            errors["missing_file"].append(mel_path)
            continue

        # 3. Load tensors
        try:
            emb = torch.load(emb_path, weights_only=True)
            mel = torch.load(mel_path, weights_only=True)
        except Exception as e:
            errors["load_failed"].append(f"{base_id}: {e}")
            continue

        # 4. Shape checks
        if emb.shape != torch.Size([256]):
            errors["bad_emb_shape"].append(f"{base_id} → {tuple(emb.shape)}")

        if mel.ndim != 2 or mel.shape[0] != 80:
            errors["bad_mel_shape"].append(f"{base_id} → {tuple(mel.shape)}")
        elif mel.shape[1] < MIN_MEL_LEN:
            errors["mel_too_short"].append(f"{base_id} → T={mel.shape[1]}")

        # 5. NaN / Inf — both tensors
        if torch.isnan(emb).any() or torch.isinf(emb).any():
            errors["corrupt_emb"].append(base_id)

        if torch.isnan(mel).any() or torch.isinf(mel).any():
            errors["corrupt_mel"].append(base_id)

    # --- Report ---
    divider = "=" * 50
    logger.info(divider)
    logger.info(f"VERIFICATION COMPLETE — sampled {len(sample)}/{len(lines)} entries")
    logger.info(divider)

    total_errors = sum(len(v) for v in errors.values())
    if total_errors == 0:
        logger.info("✅ No issues found. Dataset looks clean.")
    else:
        logger.warning(f"⚠️  {total_errors} issue(s) across {len(errors)} category/categories:")
        labels = {
            "malformed":     "Malformed lines (wrong field count)",
            "empty_text":    "Empty/whitespace text",
            "missing_file":  "Missing .pt files on disk",
            "load_failed":   "Tensor load failures",
            "bad_emb_shape": "Wrong embedding shape (expected [256])",
            "bad_mel_shape": "Wrong mel shape (expected [80, T])",
            "mel_too_short": f"Mel too short (< {MIN_MEL_LEN} frames)",
            "corrupt_emb":   "NaN/Inf in embeddings",
            "corrupt_mel":   "NaN/Inf in mel spectrograms",
        }
        for key, items in errors.items():
            label = labels.get(key, key)
            logger.warning(f"  {label}: {len(items)}")
            for item in items[:3]:       # show up to 3 examples per category
                logger.warning(f"    → {item}")
            if len(items) > 3:
                logger.warning(f"    ... and {len(items) - 3} more")

    logger.info(divider)


@app.local_entrypoint()
def main():
    verify_dataset.remote()
    
if __name__ == "__main__":
    main()