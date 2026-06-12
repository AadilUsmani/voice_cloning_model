import logging
import os
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import torch

# ── Configuration ──────────────────────────────────────────────────────────────
VOLUME_NAME    = "libritts-volume"
REMOTE_DIRS    = ["tacotron2_attention_plots", "tacotron2_inference_samples"]
LOCAL_ROOT     = Path("./training_progress")
ORGANIZED_ROOT = LOCAL_ROOT / "steps"
SYNC_INTERVAL  = 900   # seconds (15 min)
MODAL_TIMEOUT  = 120   # seconds before subprocess is killed

# ── Logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Data model ─────────────────────────────────────────────────────────────────
@dataclass
class SyncStats:
    moved:   int = 0
    plotted: int = 0
    errors:  int = 0
    skipped: int = 0   # already-organized files

    def __str__(self) -> str:
        return (
            f"📁 Sorted: {self.moved} | "
            f"🎨 Plotted: {self.plotted} | "
            f"⏭️  Skipped: {self.skipped} | "
            f"❌ Errors: {self.errors}"
        )


# ── Cloud Sync ─────────────────────────────────────────────────────────────────
def sync_from_cloud() -> bool:
    """
    Pull all progress folders from the Modal Volume to LOCAL_ROOT.

    Returns:
        True if every folder synced successfully, False if any failed.
    """
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    all_ok = True

    for folder in REMOTE_DIRS:
        log.info("Syncing '%s' from Modal volume …", folder)
        try:
            result = subprocess.run(
                ["modal", "volume", "get", VOLUME_NAME, folder, str(LOCAL_ROOT), "--force"],
                timeout=MODAL_TIMEOUT,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            if result.returncode != 0:
                log.warning("modal volume get failed for '%s':\n%s", folder, result.stderr.strip())
                all_ok = False
        except subprocess.TimeoutExpired:
            log.error("Sync timed out for '%s' after %ds.", folder, MODAL_TIMEOUT)
            all_ok = False
        except FileNotFoundError:
            log.error("'modal' CLI not found. Is it installed and on your PATH?")
            all_ok = False

    return all_ok


# ── File Organization ──────────────────────────────────────────────────────────
def _extract_step(filename: str) -> Optional[str]:
    """Return the zero-padded step number from a filename, or None."""
    match = re.search(r"step_(\d+)", filename)
    return match.group(1).zfill(6) if match else None   # zero-pad → natural sort


def _move_file(src: Path, dst: Path) -> bool:
    """
    Move src → dst atomically. Returns True on success.
    Skips silently if dst already exists (idempotent).
    """
    if dst.exists():
        return False                        # already organized in a prior run
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dst))         # atomic rename on same FS
    return True


# ── Visualization ──────────────────────────────────────────────────────────────
def _visualize_mel(pt_path: Path) -> bool:
    """
    Render a mel-spectrogram .pt tensor to a paired _vis.png.
    Uses a temp file + atomic rename to prevent corrupt half-writes.
    Returns True on success.
    """
    img_path = pt_path.with_suffix("").with_name(pt_path.stem + "_vis.png")
    if img_path.exists():
        return False                        # already visualized

    tmp_path = img_path.with_suffix(".tmp.png")

    try:
        mel = torch.load(pt_path, map_location="cpu", weights_only=False)
        mel = mel.squeeze().detach().float().numpy()

        if mel.ndim != 2:
            log.warning("Unexpected tensor shape %s in %s — skipping.", mel.shape, pt_path.name)
            return False

        fig, ax = plt.subplots(figsize=(10, 4))
        img = ax.imshow(mel, aspect="auto", origin="lower", cmap="viridis")
        ax.set_title(f"Mel Spectrogram · {pt_path.name}", fontsize=10)
        ax.set_xlabel("Frames")
        ax.set_ylabel("Mel bins")
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        fig.tight_layout()

        fig.savefig(tmp_path, dpi=120)
        plt.close(fig)

        os.replace(tmp_path, img_path)      # atomic even across drives (Python 3.3+)
        return True

    except Exception as e:
        log.error("Failed to visualize %s: %s", pt_path.name, e)
        tmp_path.unlink(missing_ok=True)    # clean up any partial write
        return False


# ── Main Processing Loop ───────────────────────────────────────────────────────
def process_new_files() -> SyncStats:
    """
    1. Walk raw sync dirs → move files into steps/step_XXXXXX/
    2. Visualize any newly moved .pt files.
    """
    stats    = SyncStats()
    pt_queue: list[Path] = []

    for folder in REMOTE_DIRS:
        raw_dir = LOCAL_ROOT / folder
        if not raw_dir.is_dir():
            continue

        for src in raw_dir.iterdir():
            if src.is_dir():
                continue                    # skip nested dirs Modal may create

            step = _extract_step(src.name)
            if step is None:
                log.debug("No step number in filename '%s' — skipping.", src.name)
                stats.skipped += 1
                continue

            dst = ORGANIZED_ROOT / f"step_{step}" / src.name
            moved = _move_file(src, dst)

            if moved:
                stats.moved += 1
                if src.suffix == ".pt":
                    pt_queue.append(dst)
            else:
                stats.skipped += 1

    for pt_path in pt_queue:
        ok = _visualize_mel(pt_path)
        if ok:
            stats.plotted += 1
        else:
            stats.errors += 1

    return stats


# ── Entry Point ────────────────────────────────────────────────────────────────
def main() -> None:
    log.info("🚀 Tacotron2 Local Monitor started.")
    log.info("   Volume  : %s", VOLUME_NAME)
    log.info("   Interval: %ds  |  Local root: %s", SYNC_INTERVAL, LOCAL_ROOT.resolve())
    log.info("   Press Ctrl+C to stop.\n")

    cycle = 0
    try:
        while True:
            cycle += 1
            log.info("── Cycle %d ──────────────────────────────────", cycle)

            synced = sync_from_cloud()
            if not synced:
                log.warning("Partial sync — some folders may be stale.")

            stats = process_new_files()
            log.info("Cycle %d done → %s", cycle, stats)
            log.info("Next sync in %ds …\n", SYNC_INTERVAL)

            time.sleep(SYNC_INTERVAL)

    except KeyboardInterrupt:
        log.info("🛑 Monitor stopped safely after %d cycle(s).", cycle)


if __name__ == "__main__":
    main()