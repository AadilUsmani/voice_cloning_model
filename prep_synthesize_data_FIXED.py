import os
import glob
import torch
import modal
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
PHASE2_NPY_DIR = "/data/melspectrograms"
OUTPUT_DIR = "/data/synthesizer_dataset"

@app.function(image=image, volumes={"/data": volume}, timeout=7200, gpu="T4")
def prepare_dataset():
    import librosa
    import torchaudio
    import numpy as np  # FIXED: Added numpy import inside function
    
    device = torch.device("cuda")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("📦 Loading Speaker Encoder model...")
    from model import SpeakerEncoder 
    model = SpeakerEncoder().to(device)
    
    # Load checkpoint and remove classifier layers
    state_dict = torch.load("/data/checkpoints/encoder_best.pth", map_location=device)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    state_dict.pop("classifier.weight", None)
    state_dict.pop("classifier.bias", None)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✅ Encoder loaded successfully")

    # Build Tacotron mel pipeline (80-bin at 22.05kHz)
    print("🔧 Building mel-spectrogram pipeline...")
    mel_transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=22050, n_fft=1024, win_length=1024, hop_length=256, n_mels=80
    ).to(device)
    print("✅ Pipeline ready")

    # --- STEP 1: INDEX EVERYTHING ONCE ---
    print("🔄 Syncing volume...")
    volume.reload()
    
    print("🔍 Scanning for WAV files...")
    wav_files = glob.glob(f"{RAW_WAV_DIR}/**/*.wav", recursive=True)
    print(f"   Found {len(wav_files)} WAV files")
    
    print("🔍 Scanning for completed embeddings...")
    existing_embeds = glob.glob(f"{OUTPUT_DIR}/*_embed.pt")
    completed_ids = {Path(f).name.replace("_embed.pt", "") for f in existing_embeds}
    print(f"   Found {len(completed_ids)} already completed")
    
    print("🔍 Indexing Phase 2 mel-spectrograms...")
    all_npy_files = glob.glob(f"{PHASE2_NPY_DIR}/**/*.npy", recursive=True)
    npy_map = {}
    for npy_file in all_npy_files:
        # Extract base identifier from filename (handles chunks like speaker_chunk_5)
        name = Path(npy_file).stem
        # Store by base name for quick lookup
        npy_map[name] = npy_file
    print(f"   Indexed {len(npy_map)} NPY files")
    
    # Load existing metadata
    meta_path = f"{OUTPUT_DIR}/train.txt"
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            existing_metadata = f.read().strip()
    else:
        existing_metadata = ""
    
    print(f"\n🚀 STARTING PROCESSING")
    print(f"   Total WAV files: {len(wav_files)}")
    print(f"   Already completed: {len(completed_ids)}")
    print(f"   Remaining: {len(wav_files) - len(completed_ids)}")
    print("=" * 60)

    # --- STEP 2: PROCESS FILES ---
    processed_count = 0
    skipped_count = 0
    error_count = 0
    new_metadata_lines = []

    for idx, wav_path in enumerate(wav_files):
        base_id = Path(wav_path).stem
        
        # INSTANT SKIP: Already done
        if base_id in completed_ids:
            continue
            
        # File paths
        emb_path = f"{OUTPUT_DIR}/{base_id}_embed.pt"
        mel_path = f"{OUTPUT_DIR}/{base_id}_mel.pt"
        txt_path = wav_path.replace(".wav", ".normalized.txt")
        
        # Check for text file
        if not os.path.exists(txt_path):
            skipped_count += 1
            continue
        
        # Find matching NPY file (Phase 2 preprocessed mel)
        # Look for exact match or chunk match
        matching_npy = None
        if base_id in npy_map:
            matching_npy = npy_map[base_id]
        else:
            # Try to find chunk files
            for npy_name, npy_path_full in npy_map.items():
                if base_id in npy_name or npy_name in base_id:
                    matching_npy = npy_path_full
                    break
        
        if not matching_npy:
            skipped_count += 1
            continue
        
        try:
            # === THE WHO: Speaker Embedding (256D from 40-bin mel) ===
            mel_40 = np.load(matching_npy)
            # Transpose from (40, T) to (T, 40) and add batch dimension
            mel_40_t = torch.from_numpy(mel_40).float().T.unsqueeze(0).to(device)
            
            with torch.no_grad():
                emb = model(mel_40_t).squeeze(0).cpu()  # [256]
            
            # === THE WHAT: Mel-Spectrogram (80-bin from raw audio) ===
            wav, _ = librosa.load(wav_path, sr=22050, mono=True)
            wav_t = torch.from_numpy(wav).unsqueeze(0).to(device)
            mel_80 = mel_transform(wav_t).squeeze(0).cpu()  # [80, T]
            
            # === THE LINK: Text transcription ===
            text = Path(txt_path).read_text(encoding="utf-8").strip()
            
            # Save the triplet
            torch.save(emb, emb_path)
            torch.save(mel_80, mel_path)
            new_metadata_lines.append(f"{base_id}|{text}|{emb_path}|{mel_path}")
            
            processed_count += 1
            completed_ids.add(base_id)
            
            # Progress logging
            if processed_count % 100 == 0:
                print(f"📊 Processed: {processed_count} new | Total: {len(completed_ids)}/{len(wav_files)} | Errors: {error_count}")
            
            # Periodic save
            if processed_count % 500 == 0:
                print(f"💾 Saving checkpoint at {processed_count} files...")
                # Append new metadata to existing
                combined_metadata = existing_metadata
                if new_metadata_lines:
                    if combined_metadata:
                        combined_metadata += "\n" + "\n".join(new_metadata_lines)
                    else:
                        combined_metadata = "\n".join(new_metadata_lines)
                
                with open(meta_path, "w", encoding="utf-8") as f:
                    f.write(combined_metadata)
                volume.commit()
                print(f"✅ Checkpoint saved and committed")
        
        except Exception as e:
            error_count += 1
            print(f"❌ Error processing {base_id}: {str(e)[:100]}")
            # CONTINUE instead of BREAK to keep processing other files
            continue
    
    # === FINAL SAVE ===
    print("\n" + "=" * 60)
    print("💾 Writing final metadata...")
    
    # Combine existing and new metadata
    combined_metadata = existing_metadata
    if new_metadata_lines:
        if combined_metadata:
            combined_metadata += "\n" + "\n".join(new_metadata_lines)
        else:
            combined_metadata = "\n".join(new_metadata_lines)
    
    with open(meta_path, "w", encoding="utf-8") as f:
        f.write(combined_metadata)
    
    print("💾 Final volume commit...")
    volume.commit()
    
    print("=" * 60)
    print("✅ PHASE 4 COMPLETE")
    print(f"📊 STATISTICS:")
    print(f"   New files processed: {processed_count}")
    print(f"   Total completed: {len(completed_ids)}")
    print(f"   Skipped (missing text/npy): {skipped_count}")
    print(f"   Errors: {error_count}")
    print(f"   Metadata entries: {len(combined_metadata.splitlines())}")
    print(f"   Output directory: {OUTPUT_DIR}")
    print("=" * 60)

@app.local_entrypoint()
def main():
    prepare_dataset.remote()

if __name__ == "__main__":
    main()
