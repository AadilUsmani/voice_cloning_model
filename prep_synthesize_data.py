import os
import glob
import torch
import modal
import numpy as np
from pathlib import Path

image = (
    modal.Image.debian_slim()
    .apt_install("ffmpeg")
    .pip_install("torch", "torchaudio", "numpy", "librosa", "soundfile", "tqdm")
    .add_local_file("model.py", remote_path="/root/model.py")
)

app = modal.App("fyp-phase4-resume")
volume = modal.Volume.from_name("libritts-volume")

RAW_WAV_DIR = "/data/LibriTTS/train-clean-100"
PHASE2_NPY_DIR = "/data/melspectrograms"
OUTPUT_DIR = "/data/synthesizer_dataset"

@app.function(image=image, volumes={"/data": volume}, timeout=7200, gpu="T4")
def prepare_dataset():
    import librosa
    import torchaudio
    
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Load Encoder
    from model import SpeakerEncoder 
    model = SpeakerEncoder().to(device)
    state_dict = torch.load("/data/checkpoints/encoder_best.pth", map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()

    # 2. Mel Pipeline
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
    ).to(device)

    # --- THE INDEXING PHASE (Do it ONCE, in memory) ---
    print("🔄 Forcing volume sync...")
    volume.reload()
    
    wav_files = glob.glob(f"{RAW_WAV_DIR}/**/*.wav", recursive=True)
    
    print("🔍 Indexing completed files...")
    existing_files = glob.glob(f"{OUTPUT_DIR}/*_embed.pt")
    completed_ids = {Path(f).name.replace("_embed.pt", "") for f in existing_files}
    
    print("🔍 Indexing Phase 2 NPY files... (This takes a few seconds)")
    all_npy_files = glob.glob(f"{PHASE2_NPY_DIR}/**/*.npy", recursive=True)
    # Create an instant dictionary lookup mapping filename -> full path
    npy_map = {Path(f).name: f for f in all_npy_files}
        
    print(f"🚀 Found {len(wav_files)} total files. Already completed: {len(completed_ids)}. Resuming...")
    # ------------------------------------------------------

    processed_count = 0
    meta_path = f"{OUTPUT_DIR}/train.txt"
    metadata = []
    
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            metadata = f.read().splitlines()

    for wav_path in wav_files:
        base_id = Path(wav_path).stem
        
        if base_id in completed_ids:
            continue 
            
        emb_path = f"{OUTPUT_DIR}/{base_id}_embed.pt"
        mel_path = f"{OUTPUT_DIR}/{base_id}_mel.pt"
        txt_path = wav_path.replace(".wav", ".normalized.txt")

        # INSTANT O(1) MEMORY LOOKUP instead of rglob
        matching_npy = [f for name, f in npy_map.items() if base_id in name]
        
        if not matching_npy or not os.path.exists(txt_path):
            continue
            
        npy_path = matching_npy[0]
            
        try:
            # Identity Embedding
            mel_40 = np.load(npy_path)
            mel_40_t = torch.from_numpy(mel_40).float().to(device).T.unsqueeze(0) 
            with torch.no_grad():
                emb = model(mel_40_t).cpu()

            # Target Spectrogram
            wav, _ = librosa.load(wav_path, sr=22050)
            wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
            mel_80 = mel_transform(wav_t).squeeze(0).cpu()

            # Save
            text = Path(txt_path).read_text().strip()
            torch.save(emb, emb_path)
            torch.save(mel_80, mel_path)
            metadata.append(f"{base_id}|{text}|{emb_path}|{mel_path}")

            processed_count += 1
            completed_ids.add(base_id) 

            if processed_count % 100 == 0:
                print(f"📊 Progress: {len(completed_ids)}/{len(wav_files)} total | New this run: {processed_count}")
            
            if processed_count % 500 == 0:
                with open(meta_path, "w") as f:
                    f.write("\n".join(metadata))
                volume.commit()

        except Exception as e:
            print(f"❌ ERROR on file {base_id}: {e}")
            break

    with open(meta_path, "w") as f:
        f.write("\n".join(metadata))
    
    volume.commit()
    print("✅ Run Complete.")

@app.local_entrypoint()
def main():
    prepare_dataset.remote()