import logging
import os

import modal

# ── Constants ──────────────────────────────────────────────────────────────────
DATASET_DIR   = "/data/synthesizer_dataset"
TRAIN_PATH    = f"{DATASET_DIR}/train.txt"
VAL_PATH      = f"{DATASET_DIR}/val.txt"
VAL_SIZE      = 200          # lines reserved for validation
MIN_TRAIN_SIZE = 500         # guard: refuse to split a tiny dataset

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Modal setup ────────────────────────────────────────────────────────────────
app    = modal.App("data-splitter")
volume = modal.Volume.from_name("libritts-volume")


# ── Core function ──────────────────────────────────────────────────────────────
@app.function(
    volumes={"/data": volume},
    timeout=300,
)
def split_file(val_size: int = VAL_SIZE, force: bool = False) -> dict[str, int]:
    """
    Split train.txt into train/val sets by slicing the last `val_size` lines.

    Args:
        val_size: Number of lines to reserve for validation.
        force:    If True, re-splits even if val.txt already exists.

    Returns:
        Dict with 'train' and 'val' line counts.
    """
    # ── Guard: idempotency ─────────────────────────────────────────────────────
    if os.path.exists(VAL_PATH) and not force:
        log.info("val.txt already exists — skipping split. Pass force=True to redo.")
        return {}

    # ── Guard: source file ─────────────────────────────────────────────────────
    if not os.path.exists(TRAIN_PATH):
        raise FileNotFoundError(f"train.txt not found at: {TRAIN_PATH}")

    # ── Read ───────────────────────────────────────────────────────────────────
    log.info("Reading %s …", TRAIN_PATH)
    with open(TRAIN_PATH, "r", encoding="utf-8") as f:
        lines = f.readlines()

    total = len(lines)
    log.info("Total lines: %d", total)

    # ── Guard: dataset size ────────────────────────────────────────────────────
    if total < val_size + MIN_TRAIN_SIZE:
        raise ValueError(
            f"Dataset too small ({total} lines) to safely reserve "
            f"{val_size} for val and keep ≥{MIN_TRAIN_SIZE} for train."
        )

    train_lines = lines[:-val_size]
    val_lines   = lines[-val_size:]

    # ── Write ──────────────────────────────────────────────────────────────────
    log.info("Writing train.txt  (%d lines) …", len(train_lines))
    with open(TRAIN_PATH, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    log.info("Writing val.txt    (%d lines) …", len(val_lines))
    with open(VAL_PATH, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    # ── Commit to persistent volume ────────────────────────────────────────────
    volume.commit()
    log.info("Volume committed.")

    counts = {"train": len(train_lines), "val": len(val_lines)}
    log.info("Done ✅  | train=%d  val=%d", counts["train"], counts["val"])
    return counts


# ── Local entrypoint ───────────────────────────────────────────────────────────
@app.local_entrypoint()
def main() -> None:
    result = split_file.remote(val_size=VAL_SIZE, force=False)
    if result:
        print(f"\n📊 Split complete → train: {result['train']} | val: {result['val']}")