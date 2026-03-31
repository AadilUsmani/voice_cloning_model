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
    .pip_install(
        "torch",
        "torchcodec",
        "torchaudio",
        "numpy",
        "librosa",
        "soundfile",
        "tqdm"
    )
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


def build_mel_pipeline(
    sample_rate: int,
    n_fft: int,
    win_length: int,
    hop_length: int,
    n_mels: int,
    f_max: int | None = None,
    power: float = 2.0,
    normalized: bool = False,
    device: torch.device = torch.device("cpu"),
) -> torch.nn.Sequential:
    """Builds a reusable MelSpectrogram + AmplitudeToDB pipeline on the given device."""
    return torch.nn.Sequential(
        torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            f_max=f_max,
            n_mels=n_mels,
            power=power,
            normalized=normalized,
        ),
        torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80),
    ).to(device)


def load_audio(wav_path: str, target_sr: int, device: torch.device) -> torch.Tensor:
    """Loads a wav file, resamples if needed, returns a [1, T] tensor on device."""
    wav, sr = torchaudio.load(wav_path)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
    # Mix down to mono if stereo
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)
    return wav.to(device)


@app.function(image=image, volumes={"/data": volume}, timeout=3600, gpu="T4")
def prepare_dataset():
    import torchaudio
    import torchaudio.transforms
    import torchaudio.functional
    
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    from model import SpeakerEncoder

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Starting preparation on {device}...")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- Helper function with torchaudio access ---
    def build_mel_pipeline_local(
        sample_rate: int,
        n_fft: int,
        win_length: int,
        hop_length: int,
        n_mels: int,
        f_max: int | None = None,
        power: float = 2.0,
        normalized: bool = False,
    ) -> torch.nn.Sequential:
        """Builds a reusable MelSpectrogram + AmplitudeToDB pipeline on the given device."""
        return torch.nn.Sequential(
            torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                f_max=f_max,
                n_mels=n_mels,
                power=power,
                normalized=normalized,
            ),
            torchaudio.transforms.AmplitudeToDB(stype="magnitude", top_db=80),
        ).to(device)

    def load_audio_local(wav_path: str, target_sr: int) -> torch.Tensor:
        """Loads a wav file, resamples if needed, returns a [1, T] tensor on device."""
        wav, sr = torchaudio.load(wav_path)
        if sr != target_sr:
            wav = torchaudio.functional.resample(wav, orig_freq=sr, new_freq=target_sr)
        # Mix down to mono if stereo
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.to(device)

    # --- Load Encoder (once) ---
    logger.info("Loading encoder checkpoint...")
    model = SpeakerEncoder(n_speakers=N_SPEAKERS).to(device)
    raw_sd = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    clean_sd = {k.replace("_orig_mod.", ""): v for k, v in raw_sd.items()}
    model.load_state_dict(clean_sd, strict=False)
    model.eval()

    # --- Build Mel Pipelines (once, reused every iteration) ---
    enc_mel_pipeline  = build_mel_pipeline_local(
        ENC_SAMPLE_RATE, ENC_N_FFT, ENC_WIN_LENGTH, ENC_HOP_LENGTH, ENC_N_MELS
    )
    taco_mel_pipeline = build_mel_pipeline_local(
        TACO_SAMPLE_RATE, TACO_N_FFT, TACO_WIN_LENGTH, TACO_HOP_LENGTH, TACO_N_MELS,
        f_max=8000, power=1.0, normalized=True
    )

    # --- Scan Files ---
    wav_files = glob.glob(f"{RAW_LIBRITTS_PATH}/**/*.wav", recursive=True)
    logger.info(f"Found {len(wav_files)} audio files.")

    metadata = []
    processed_count = 0

    with torch.no_grad():
        for wav_path in wav_files:
            txt_path = wav_path.replace(".wav", ".normalized.txt")
            if not os.path.exists(txt_path):
                continue

            text = Path(txt_path).read_text(encoding="utf-8").strip()
            if len(text) < 2:
                continue

            try:
                # A. Speaker embedding (256D)
                wav_enc = load_audio_local(wav_path, ENC_SAMPLE_RATE)
                mel_enc = enc_mel_pipeline(wav_enc)           # [1, 40, T]
                output  = model(mel_enc)
                embedding = (output[0] if isinstance(output, tuple) else output)
                embedding = embedding.squeeze(0).cpu()        # [256]

                # B. Tacotron target mel (80D)
                wav_taco = load_audio_local(wav_path, TACO_SAMPLE_RATE)
                mel_taco = taco_mel_pipeline(wav_taco).squeeze(0).cpu()  # [80, T]

                # C. Save artifacts
                base      = Path(wav_path).stem
                emb_path  = os.path.join(OUTPUT_DIR, f"{base}_embed.pt")
                mel_path  = os.path.join(OUTPUT_DIR, f"{base}_mel.pt")

                torch.save(embedding, emb_path)
                torch.save(mel_taco,  mel_path)

                metadata.append(f"{base}|{text}|{emb_path}|{mel_path}")
                processed_count += 1

                if processed_count % LOG_INTERVAL == 0:
                    logger.info(f"Processed {processed_count}/{len(wav_files)} files...")
                    volume.commit()

            except Exception as e:
                logger.warning(f"Skipping {wav_path}: {e}")

    # --- Write Metadata ---
    meta_path = os.path.join(OUTPUT_DIR, "train_metadata.txt")
    Path(meta_path).write_text("\n".join(metadata), encoding="utf-8")
    volume.commit()

    logger.info(f"✅ Done. Saved {processed_count} triplets to {OUTPUT_DIR}")


@app.local_entrypoint()
def main():
    prepare_dataset.remote()

if __name__ == "__main__":
    main()