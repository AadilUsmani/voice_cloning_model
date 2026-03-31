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

    # 2. Build the 80-bin Mel Pipeline (Target for Tacotron)
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

    # 3. Scan for Raw Audio (To get Text + 80-bin Mel)
    wav_files = glob.glob(f"{RAW_LIBRITTS_PATH}/**/*.wav", recursive=True)
    logger.info(f"Found {len(wav_files)} audio files.")
    
    metadata = []
    processed_count = 0

    with torch.no_grad():
        for wav_path in wav_files:
            base = Path(wav_path).stem
            txt_path = wav_path.replace(".wav", ".normalized.txt")
            
            # Look for the Phase 2 Preprocessed 40-bin Mel (for the Encoder)
            # Extract speaker ID from path: /data/LibriTTS/train-clean-100/{speaker_id}/{chapter_id}/{base}.wav
            path_parts = Path(wav_path).parts
            speaker_id = path_parts[-3]  # Get speaker_id from directory structure
            npy_path = f"/data/melspectrograms/{speaker_id}/{base}.npy"
            
            if not os.path.exists(txt_path) or not os.path.exists(npy_path):
                continue

            try:
                # --- A. ENCODER PASS (IDENTITY) ---
                # Load the 40-bin mel from Phase 2
                mel_40 = np.load(npy_path)
                mel_40_tensor = torch.from_numpy(mel_40).float().unsqueeze(0).to(device) # [1, 40, T]
                
                # CRITICAL FIX: Transpose from [1, 40, T] to [1, T, 40] for the LSTM
                mel_40_input = mel_40_tensor.permute(0, 2, 1) 
                
                output = model(mel_40_input)
                embedding = (output[0] if isinstance(output, tuple) else output).squeeze(0).cpu()

                # --- B. TACOTRON PASS (TARGET SPEECH) ---
                wav, sr = torchaudio.load(wav_path)
                if sr != TACO_SAMPLE_RATE:
                    wav = torchaudio.functional.resample(wav, sr, TACO_SAMPLE_RATE)
                if wav.shape[0] > 1:
                    wav = wav.mean(dim=0, keepdim=True)
                mel_80 = taco_mel_pipeline(wav.to(device)).squeeze(0).cpu()

                # --- C. SAVE ---
                text = Path(txt_path).read_text(encoding="utf-8").strip()
                emb_p = os.path.join(OUTPUT_DIR, f"{base}_embed.pt")
                mel_p = os.path.join(OUTPUT_DIR, f"{base}_mel.pt")
                
                torch.save(embedding, emb_p)
                torch.save(mel_80, mel_p)
                metadata.append(f"{base}|{text}|{emb_p}|{mel_p}")
                
                processed_count += 1
                if processed_count % LOG_INTERVAL == 0:
                    logger.info(f"Processed {processed_count}/{len(wav_files)} files...")
                    volume.commit()

            except Exception as e:
                logger.warning(f"Error on {base}: {e}")

    # Write metadata
    meta_path = os.path.join(OUTPUT_DIR, "train_metadata.txt")
    Path(meta_path).write_text("\n".join(metadata), encoding="utf-8")
    volume.commit()
    logger.info(f"✅ Done. Saved {processed_count} triplets to {OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    prepare_dataset.remote()

if __name__ == "__main__":
    main()
