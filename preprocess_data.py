import os
import json
import logging
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed

import modal

# --- Modal Setup ---
app = modal.App("fyp-voice-clone")
volume = modal.Volume.from_name("libritts-volume")
image = modal.Image.debian_slim().pip_install("librosa", "numpy", "soundfile", "tqdm")

# --- Constants ---
INPUT_DIR = "/data/LibriTTS/train-clean-100"
OUTPUT_DIR = "/data/melspectrograms"
COMMIT_EVERY = 20  # checkpoint to volume every N completed speakers
AUDIO_PARAMS = {
    "sample_rate": 16000,
    "n_mels": 40,
    "chunk_duration": 1.6,
    "n_fft": 800,
    "hop_length": 200,
}


def process_speaker(speaker_path: str, output_dir: str, params: dict) -> tuple[str, int, str | None]:
    """Process a single speaker: extract mel-spectrograms from all WAV chunks."""
    import os
    import glob
    import librosa
    import numpy as np

    speaker_id = os.path.basename(speaker_path)
    speaker_out_dir = os.path.join(output_dir, speaker_id)

    # Resume: skip already-processed speakers
    if os.path.exists(speaker_out_dir) and os.listdir(speaker_out_dir):
        existing = len(glob.glob(f"{speaker_out_dir}/*.npy"))
        return speaker_id, existing, None

    os.makedirs(speaker_out_dir, exist_ok=True)

    sr = params["sample_rate"]
    samples_per_chunk = int(sr * params["chunk_duration"])
    chunk_counter = 0

    for audio_path in glob.glob(f"{speaker_path}/*/*.wav"):
        try:
            wav, _ = librosa.load(audio_path, sr=sr)
            wav, _ = librosa.effects.trim(wav, top_db=30)

            for i in range(0, len(wav) - samples_per_chunk, samples_per_chunk):
                chunk = wav[i : i + samples_per_chunk]
                mel = librosa.feature.melspectrogram(
                    y=chunk,
                    sr=sr,
                    n_mels=params["n_mels"],
                    n_fft=params["n_fft"],
                    hop_length=params["hop_length"],
                )
                mel_db = librosa.power_to_db(mel, ref=np.max)
                save_path = os.path.join(speaker_out_dir, f"{speaker_id}_chunk_{chunk_counter}.npy")
                np.save(save_path, mel_db)
                chunk_counter += 1

        except Exception as e:
            return speaker_id, 0, f"Error on {audio_path}: {e}"

    return speaker_id, chunk_counter, None


@app.function(image=image, volumes={"/data": volume}, timeout=7200, cpu=4.0)
def preprocess_dataset():
    from tqdm import tqdm

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    logger = logging.getLogger(__name__)

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    speaker_folders = glob.glob(f"{INPUT_DIR}/*")

    if not speaker_folders:
        logger.error("No speaker folders found. Check Phase 1 extraction.")
        return

    logger.info(f"Found {len(speaker_folders)} speakers. Starting parallel processing...")

    metadata: dict[str, int] = {}

    with ProcessPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(process_speaker, path, OUTPUT_DIR, AUDIO_PARAMS): path
            for path in speaker_folders
        }
        for i, future in enumerate(tqdm(as_completed(futures), total=len(speaker_folders), desc="Processing Speakers"), 1):
            speaker_id, count, error = future.result()
            if error:
                logger.warning(f"Speaker {speaker_id}: {error}")
            if count > 0:
                metadata[speaker_id] = count
            if i % COMMIT_EVERY == 0:
                logger.info(f"Checkpointing at {i}/{len(speaker_folders)} speakers...")
                volume.commit()

    metadata_path = os.path.join(OUTPUT_DIR, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)
    logger.info(f"Metadata saved → {metadata_path}")

    volume.commit()
    logger.info(f"Phase 2 complete. Total chunks: {sum(metadata.values())}")


@app.local_entrypoint()
def main():
    preprocess_dataset.remote()


if __name__ == "__main__":
    main()