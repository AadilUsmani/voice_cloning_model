''' 
import modal
import os

app = modal.App("check-data")
volume = modal.Volume.from_name("libritts-volume")

@app.function(volumes={"/data": volume})
def count_files():
    path = "/data/LibriTTS/train-clean-100"
    speakers = [d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))]
    
    total_wavs = 0
    for root, _, files in os.walk(path):
        total_wavs += sum(1 for f in files if f.endswith(".wav"))
    
    print(f"📊 Dataset Statistics:")
    print(f"Number of Speakers: {len(speakers)}")
    print(f"Total .wav Files: {total_wavs}")

@app.local_entrypoint()
def main():
    count_files.remote()

if __name__ == "__main__":
    main()

'''
import modal
import glob

app = modal.App("inspect-data")
volume = modal.Volume.from_name("libritts-volume")

# 1. We must tell the cloud container to install numpy
image = modal.Image.debian_slim().pip_install("numpy")

@app.function(image=image, volumes={"/data": volume})
def check_mel():
    # 2. Import numpy inside the cloud function
    import numpy as np 
    
    files = glob.glob("/data/melspectrograms/*/*.npy")
    if not files:
        print("No files found! Check your folder paths.")
        return
    
    sample_file = files[0]
    mel_array = np.load(sample_file)
    
    print(f"📄 File: {sample_file}")
    print(f"📐 Shape (Mel Bands x Time Steps): {mel_array.shape}")
    print(f"🔢 Max Volume (dB): {np.max(mel_array):.2f}")
    print(f"🔢 Min Volume (dB): {np.min(mel_array):.2f}")

@app.local_entrypoint()
def main():
    check_mel.remote()

if __name__ == "__main__":
    main()