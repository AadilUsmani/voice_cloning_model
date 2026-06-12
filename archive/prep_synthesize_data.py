import os
import glob
import torch
import modal
import librosa
from pathlib import Path

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "numpy", "librosa", "soundfile", "tqdm")
    .add_local_file("model.py", remote_path="/root/model.py")
)

app = modal.App("fyp-phase4-final")
volume = modal.Volume.from_name("libritts-volume")

RAW_WAV_DIR = "/data/LibriTTS/train-clean-100"
OUTPUT_DIR = "/data/synthesizer_dataset"

@app.function(image=image, volumes={"/data": volume}, timeout=10800, gpu="T4")
def prepare_dataset():
    import torchaudio
    import numpy as np
    
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Loading Speaker Encoder model...")
    from model import SpeakerEncoder 
    model = SpeakerEncoder().to(device)
    
    # Load checkpoint
    state_dict = torch.load("/data/checkpoints/encoder_best.pth", map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Encoder loaded successfully")

    print("🔧 Building dual mel-spectrogram pipelines...")
    # Pipeline for the Synthesizer (80-bin, 22.05kHz)
    mel_transform_80 = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
    ).to(device)
    
    # Pipeline for the Encoder (40-bin, 16kHz)
    mel_transform_40 = torchaudio.transforms.MelSpectrogram(
        sample_rate=16000, n_fft=1024, win_length=1024, hop_length=256, n_mels=40
    ).to(device)
    print("✅ Pipelines ready")

    print("🔄 Syncing volume...")
    volume.reload()
    
    print("🔍 Scanning for WAV files...")
    wav_files = glob.glob(f"{RAW_WAV_DIR}/**/*.wav", recursive=True)
    
    existing_embeds = glob.glob(f"{OUTPUT_DIR}/*_embed.pt")
    completed_ids = {Path(f).name.replace("_embed.pt", "") for f in existing_embeds}
    
    meta_path = f"{OUTPUT_DIR}/train.txt"
    existing_metadata = open(meta_path, "r").read().strip() if os.path.exists(meta_path) else ""
    
    print(f"\n🚀 STARTING PROCESSING")
    print(f"   Total WAV: {len(wav_files)} | Completed: {len(completed_ids)}")
    print("=" * 60)

    processed_count = 0
    skipped_count = 0
    error_count = 0
    new_metadata_lines = []

    for wav_path in wav_files:
        base_id = Path(wav_path).stem
        
        if base_id in completed_ids:
            continue
            
        emb_path = f"{OUTPUT_DIR}/{base_id}_embed.pt"
        mel_path = f"{OUTPUT_DIR}/{base_id}_mel.pt"
        txt_path = wav_path.replace(".wav", ".normalized.txt")
        
        if not os.path.exists(txt_path):
            skipped_count += 1
            continue
        
        try:
            # === OPTIMIZATION 1: Load Once, Resample Twice ===
            wav, sr = librosa.load(wav_path, sr=None, mono=True)
            wav_16k = librosa.resample(wav, orig_sr=sr, target_sr=16000)
            wav_22k = librosa.resample(wav, orig_sr=sr, target_sr=22050)
            
            wav_16k_t = torch.from_numpy(wav_16k).unsqueeze(0).to(device)
            wav_22k_t = torch.from_numpy(wav_22k).unsqueeze(0).to(device)
            
            # === THE WHO: Speaker Embedding (40-bin) ===
            mel_40 = mel_transform_40(wav_16k_t)  # Shape: [1, 40, T]
            mel_40_t = mel_40.transpose(1, 2)     # Shape: [1, T, 40]
            
            with torch.no_grad():
                emb = model(mel_40_t).squeeze(0).cpu()  # [256]
            
            # === THE WHAT: Mel-Spectrogram target (80-bin) ===
            mel_80 = mel_transform_80(wav_22k_t).squeeze(0)  # [80, T]
            
            # === OPTIMIZATION 2: Log Normalization ===
            # Synthesizers require log-mel scale to learn properly
            mel_80 = torch.log(torch.clamp(mel_80, min=1e-5)).cpu()
            
            # === THE LINK: Text transcription ===
            text = Path(txt_path).read_text(encoding="utf-8").strip()
            
            # Save the triplet
            torch.save(emb, emb_path)
            torch.save(mel_80, mel_path)
            
            # === OPTIMIZATION 3: Better Metadata Ordering ===
            new_metadata_lines.append(f"{base_id}|{text}|{mel_path}|{emb_path}")
            
            processed_count += 1
            completed_ids.add(base_id)
            
            if processed_count % 100 == 0:
                print(f"📊 Processed: {processed_count} | Total: {len(completed_ids)}/{len(wav_files)}")
            
            if processed_count % 500 == 0:
                combined_metadata = existing_metadata + "\n" + "\n".join(new_metadata_lines) if existing_metadata else "\n".join(new_metadata_lines)
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(combined_metadata.strip())
                new_metadata_lines = [] # Reset buffer
                existing_metadata = combined_metadata # Update existing
                volume.commit()
                print("💾 Checkpoint saved and committed")
                
        except Exception as e:
            error_count += 1
            print(f"❌ Error on {base_id}: {str(e)[:100]}")
            continue
    
    # Final cleanup save
    if new_metadata_lines:
        combined_metadata = existing_metadata + "\n" + "\n".join(new_metadata_lines) if existing_metadata else "\n".join(new_metadata_lines)
        with open(meta_path, "w", encoding="utf-8") as f:
            f.write(combined_metadata.strip())
    
    volume.commit()
    print("✅ PHASE 4 COMPLETE")

@app.local_entrypoint()
def main():
    prepare_dataset.remote()

if __name__ == "__main__":
    main()