import os
import glob
import torch
import modal
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# --- Modal Setup ---
app = modal.App("fyp-synth-prep")
volume = modal.Volume.from_name("libritts-volume")

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "numpy", "librosa", "soundfile", "tqdm")
    .add_local_file("model.py", remote_path="/root/model.py")
)

# --- Paths ---
RAW_LIBRITTS_PATH = "/data/LibriTTS/train-clean-100"
CHECKPOINT_PATH   = "/data/checkpoints/encoder_best.pth"
OUTPUT_DIR        = "/data/synthesizer_dataset"

# --- Encoder Mel Config (16kHz, 40-bin) ---
ENC_SAMPLE_RATE = 16000
ENC_N_FFT       = 400
ENC_HOP_LENGTH  = 160
ENC_WIN_LENGTH  = 400
ENC_N_MELS      = 40

# --- Tacotron Mel Config (22050Hz, 80-bin) ---
TACO_SAMPLE_RATE = 22050
TACO_N_FFT       = 1024
TACO_HOP_LENGTH  = 256
TACO_WIN_LENGTH  = 1024
TACO_N_MELS      = 80

N_SPEAKERS        = 245
LOG_INTERVAL      = 500
COMMIT_INTERVAL   = 500


@app.function(image=image, volumes={"/data": volume}, timeout=3600, gpu="T4")
def prepare_dataset():
    import torch
    import torchaudio
    import torchaudio.transforms
    import librosa
    import numpy as np
    from model import SpeakerEncoder

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting preparation on {device}...")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Encoder with correct input_size
    logger.info("Loading encoder checkpoint...")
    model = SpeakerEncoder(n_speakers=N_SPEAKERS).to(device)
    raw_sd = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()
    logger.info("✅ Encoder loaded successfully")

    # 2. Build BOTH mel pipelines - 40-bin for encoder, 80-bin for Tacotron
    logger.info("Building mel-spectrogram pipelines...")
    
    # 40-bin encoder pipeline (16kHz)
    enc_mel_pipeline = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=ENC_SAMPLE_RATE,
            n_fft=ENC_N_FFT,
            win_length=ENC_WIN_LENGTH,
            hop_length=ENC_HOP_LENGTH,
            n_mels=ENC_N_MELS,
            power=2.0,
            normalized=False
        ),
        torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
    ).to(device)
    
    # 80-bin Tacotron pipeline (22.05kHz)
    taco_mel_pipeline = torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=TACO_SAMPLE_RATE, 
            n_fft=TACO_N_FFT, 
            win_length=TACO_WIN_LENGTH, 
            hop_length=TACO_HOP_LENGTH, 
            n_mels=TACO_N_MELS, 
            power=1.0, 
            normalized=True
        ),
        torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80)
    ).to(device)
    logger.info("✅ Pipelines built successfully")

    # 3. Scan for Raw Audio files (we'll generate both mels on-the-fly)
    logger.info("Scanning for audio files...")
    wav_files = glob.glob(f"{RAW_LIBRITTS_PATH}/**/*.wav", recursive=True)
    logger.info(f"✅ Found {len(wav_files)} audio files.")
    
    metadata = []
    processed_count = 0
    skipped_count = 0
    error_count = 0

    logger.info("Starting processing loop...")
    with torch.no_grad():
        for idx, wav_path in enumerate(wav_files):
            base = Path(wav_path).stem
            txt_path = wav_path.replace(".wav", ".normalized.txt")
            
            if not os.path.exists(txt_path):
                skipped_count += 1
                continue

            try:
                text = Path(txt_path).read_text(encoding="utf-8").strip()
                if len(text) < 2:
                    skipped_count += 1
                    continue

                # --- A. ENCODER PASS: Generate 40-bin mel at 16kHz using LIBROSA ---
                try:
                    wav_16k, _ = librosa.load(wav_path, sr=ENC_SAMPLE_RATE, mono=True)
                    wav_16k_tensor = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
                except Exception as e:
                    logger.warning(f"Librosa failed to load {base} at 16kHz: {e}")
                    error_count += 1
                    continue
                
                mel_40 = enc_mel_pipeline(wav_16k_tensor)  # [1, 40, T]
                mel_40_input = mel_40.permute(0, 2, 1)  # [1, T, 40] for LSTM
                
                output = model(mel_40_input)
                embedding = (output[0] if isinstance(output, tuple) else output).squeeze(0).cpu()

                # --- B. TACOTRON PASS: Generate 80-bin mel at 22.05kHz using LIBROSA ---
                try:
                    wav_22k, _ = librosa.load(wav_path, sr=TACO_SAMPLE_RATE, mono=True)
                    wav_22k_tensor = torch.from_numpy(wav_22k).unsqueeze(0).to(device)
                except Exception as e:
                    logger.warning(f"Librosa failed to load {base} at 22.05kHz: {e}")
                    error_count += 1
                    continue
                    
                mel_80 = taco_mel_pipeline(wav_22k_tensor).squeeze(0).cpu()  # [80, T]

                # --- C. SAVE TRIPLETS ---
                emb_p = os.path.join(OUTPUT_DIR, f"{base}_embed.pt")
                mel_p = os.path.join(OUTPUT_DIR, f"{base}_mel.pt")
                
                torch.save(embedding, emb_p)
                torch.save(mel_80, mel_p)
                metadata.append(f"{base}|{text}|{emb_p}|{mel_p}")
                
                processed_count += 1
                
                # Log progress every 100 files
                if processed_count % 100 == 0:
                    logger.info(f"📊 Progress: {processed_count}/{len(wav_files)} processed | Skipped: {skipped_count} | Errors: {error_count}")
                
                # Commit to volume every 500 files
                if processed_count % LOG_INTERVAL == 0:
                    logger.info(f"💾 Committing to volume at {processed_count} files...")
                    volume.commit()
                    logger.info("✅ Volume committed")

            except Exception as e:
                logger.warning(f"❌ Error processing {base}: {e}")
                error_count += 1

    # Write metadata
    logger.info(f"Writing metadata file...")
    meta_path = os.path.join(OUTPUT_DIR, "train_metadata.txt")
    Path(meta_path).write_text("\n".join(metadata), encoding="utf-8")
    
    logger.info(f"Final commit to volume...")
    volume.commit()
    
    logger.info("=" * 60)
    logger.info(f"✅ PHASE 4 COMPLETE")
    logger.info(f"📁 Output directory: {OUTPUT_DIR}")
    logger.info(f"✅ Successfully processed: {processed_count} files")
    logger.info(f"⏭️  Skipped (no text): {skipped_count} files")
    logger.info(f"❌ Errors: {error_count} files")
    logger.info(f"📄 Metadata entries: {len(metadata)}")
    logger.info("=" * 60)


@app.local_entrypoint()
def main():
    prepare_dataset.remote()

if __name__ == "__main__":
    main()
