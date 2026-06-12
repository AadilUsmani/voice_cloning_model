import os
import torch
import librosa
import numpy as np

# ── NVIDIA Tacotron 2 Acoustic Hyperparameters ──
SAMPLE_RATE = 22050
N_FFT = 1024
HOP_LENGTH = 256
WIN_LENGTH = 1024
N_MELS = 80
FMIN = 0.0
FMAX = 8000.0

# ── Directories ──
USER_DIR = "user_data"
WAV_DIR = os.path.join(USER_DIR, "wavs")
MEL_DIR = os.path.join(USER_DIR, "mels")
METADATA_FILE = os.path.join(USER_DIR, "metadata.txt")
OUTPUT_LIST = os.path.join(USER_DIR, "train_list.txt")

def process_audio_to_mel(wav_path):
    """Loads audio, trims silence, and generates NVIDIA-compatible Mel-Spectrogram."""
    # 1. Load and resample to 22.05 kHz
    wav, _ = librosa.load(wav_path, sr=SAMPLE_RATE)
    
    # 2. Trim silence (CRITICAL for attention alignment)
    wav, _ = librosa.effects.trim(wav, top_db=60)
    
    # 3. Compute Mel-spectrogram
    mel = librosa.feature.melspectrogram(
        y=wav, sr=SAMPLE_RATE, n_fft=N_FFT,
        hop_length=HOP_LENGTH, win_length=WIN_LENGTH,
        n_mels=N_MELS, fmin=FMIN, fmax=FMAX
    )
    
    # 4. Dynamic range compression (Log-Mel)
    mel = np.log(np.clip(mel, a_min=1e-5, a_max=None))
    
    # 5. Convert to PyTorch Tensor (Shape: [80, Time])
    return torch.FloatTensor(mel)

def main():
    os.makedirs(MEL_DIR, exist_ok=True)
    
    if not os.path.exists(METADATA_FILE):
        print(f"❌ Error: Could not find {METADATA_FILE}. Please create it!")
        return

    train_metadata = []
    
    print("⏳ Processing User Audio...")
    with open(METADATA_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    for line in lines:
        if '|' not in line:
            continue
            
        filename, text = line.strip().split('|', 1)
        if not filename.endswith('.wav'):
            filename += '.wav'
            
        wav_path = os.path.join(WAV_DIR, filename)
        
        if not os.path.exists(wav_path):
            print(f"⚠️ Warning: Missing audio file - {filename}")
            continue
            
        # Process and save Mel tensor
        mel_tensor = process_audio_to_mel(wav_path)
        mel_filename = filename.replace('.wav', '.pt')
        mel_path = os.path.join(MEL_DIR, mel_filename)
        
        torch.save(mel_tensor, mel_path)
        
        # Add to training list (Format: path_to_mel|text)
        train_metadata.append(f"{mel_path}|{text}")
        print(f"✅ Processed: {filename} -> Shape: {list(mel_tensor.shape)}")

    # Write the final train list
    with open(OUTPUT_LIST, 'w', encoding='utf-8') as f:
        f.write('\n'.join(train_metadata))
        
    print(f"\n🎉 Done! Processed {len(train_metadata)} files.")
    print(f"👉 Ready for fine-tuning. Training list saved to: {OUTPUT_LIST}")

if __name__ == "__main__":
    main()