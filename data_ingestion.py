import modal
import os
import subprocess
import shutil
import logging
from pathlib import Path

# --- Configuration ---
app = modal.App("fyp-voice-clone")
volume = modal.Volume.from_name("libritts-volume", create_if_missing=True)
image = modal.Image.debian_slim().apt_install("wget", "tar")

# Constants (easier to tune later)
DATA_ROOT = "/data"
MIN_FREE_SPACE_GB = 25.0
WGET_RETRIES = 5
VERIFY_LIMIT = 5  # Show top N files for verification

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def setup_extraction_dir(extract_dir: str) -> bool:
    """Ensure extraction directory exists. Returns True if created, False if existed."""
    os.makedirs(extract_dir, exist_ok=True)
    return True


def check_disk_space(required_gb: float) -> None:
    """Validate sufficient free disk space."""
    total, used, free = shutil.disk_usage(DATA_ROOT)
    free_gb = free / (2**30)
    if free_gb < required_gb:
        raise OSError(
            f"Insufficient disk space: {free_gb:.2f} GB available, "
            f"{required_gb:.2f} GB required"
        )
    logger.info(f"Disk check passed: {free_gb:.2f} GB available")


def download_file(url: str, output_path: str) -> None:
    """Download file with resume support."""
    logger.info(f"Downloading from {url}...")
    subprocess.run(
        ["wget", "-c", f"--tries={WGET_RETRIES}", "-O", output_path, url],
        check=True,
        capture_output=True
    )
    logger.info(f"Download complete: {output_path}")


def extract_archive(tar_path: str, extract_to: str) -> None:
    """Extract tar.gz archive."""
    logger.info(f"Extracting {tar_path}...")
    subprocess.run(
        ["tar", "-xzf", tar_path, "-C", extract_to],
        check=True,
        capture_output=True
    )
    logger.info("Extraction complete")


def cleanup_file(file_path: str) -> None:
    """Safely remove a file."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.info(f"Cleaned up: {file_path}")
    except OSError as e:
        logger.warning(f"Failed to clean up {file_path}: {e}")


def cleanup_directory(dir_path: str) -> None:
    """Safely remove a directory tree."""
    try:
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
            logger.info(f"Cleaned up directory: {dir_path}")
    except OSError as e:
        logger.warning(f"Failed to clean up directory {dir_path}: {e}")


def verify_extraction(extracted_dir: str) -> None:
    """Quick verification of extracted contents."""
    try:
        contents = os.listdir(extracted_dir)
        logger.info(
            f"Extraction verified. Contents (top {VERIFY_LIMIT}): "
            f"{contents[:VERIFY_LIMIT]}"
        )
    except OSError as e:
        raise FileNotFoundError(f"Extraction failed: directory not accessible: {e}")


@app.function(image=image, volumes={DATA_ROOT: volume}, timeout=7200)
def download_and_extract_data(dataset_name: str, dataset_url: str) -> dict:
    """
    Download and extract a dataset to Modal volume.
    
    Args:
        dataset_name: Name of dataset (used for folder and tar naming)
        dataset_url: Full URL to tar.gz file
        
    Returns:
        dict with status and path information
    """
    tar_path = os.path.join(DATA_ROOT, f"{dataset_name}.tar.gz")
    extracted_dir = os.path.join(DATA_ROOT, "LibriTTS", dataset_name)
    
    logger.info(f"Starting data ingestion: {dataset_name}")
    
    # Early exit if already exists
    if os.path.exists(extracted_dir):
        logger.info(f"Dataset already exists: {extracted_dir}")
        return {"status": "skipped", "path": extracted_dir}
    
    # Pre-flight checks
    setup_extraction_dir(os.path.dirname(extracted_dir))
    check_disk_space(MIN_FREE_SPACE_GB)
    
    try:
        # Download and extract
        download_file(dataset_url, tar_path)
        extract_archive(tar_path, DATA_ROOT)
        
        # Validate extraction worked
        verify_extraction(extracted_dir)
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        logger.info("Initiating cleanup of corrupted files...")
        cleanup_file(tar_path)
        cleanup_directory(extracted_dir)
        raise RuntimeError(f"Data ingestion failed and rolled back: {e}")
    
    finally:
        # Always clean up the tar file (whether success or failure)
        cleanup_file(tar_path)
    
    # Persist changes
    logger.info("Committing volume changes...")
    volume.commit()
    
    logger.info(f"✅ Data ingestion complete: {dataset_name}")
    return {"status": "success", "path": extracted_dir}


@app.local_entrypoint()
def main():
    """Execute data ingestion."""
    try:
        result = download_and_extract_data.remote(
            dataset_name="train-clean-100",
            dataset_url="https://www.openslr.org/resources/60/train-clean-100.tar.gz"
        )
        logger.info(f"Result: {result}")
    except Exception as e:
        logger.error(f"Main execution failed: {e}")
        raise


if __name__ == "__main__":
    main()