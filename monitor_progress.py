import os
import subprocess
import time
import torch
import matplotlib.pyplot as plt

# --- Configuration ---
VOLUME_NAME = "libritts-volume"
REMOTE_DIRS = ["tacotron2_attention_plots", "tacotron2_inference_samples"]
LOCAL_ROOT = "./training_progress"

def sync_from_cloud():
    """Syncs all progress files from Modal Volume to your laptop."""
    if not os.path.exists(LOCAL_ROOT):
        os.makedirs(LOCAL_ROOT)
    
    for folder in REMOTE_DIRS:
        print(f"🔄 Syncing {folder}...")
        try:
            # Command: modal volume get <volume_name> <remote_path> <local_destination>
            subprocess.run([
                "modal", "volume", "get", 
                VOLUME_NAME, folder, LOCAL_ROOT,"--force"
            ], check=True)
        except Exception as e:
            print(f"⚠️ Could not sync {folder}. Error: {e}")

def convert_mels_to_images():
    """Finds .pt mel files and turns them into viewable PNGs."""
    inference_path = os.path.join(LOCAL_ROOT, "tacotron2_inference_samples")
    if not os.path.exists(inference_path):
        return

    for file in os.listdir(inference_path):
        if file.endswith(".pt"):
            img_name = file.replace(".pt", "_vis.png")
            img_path = os.path.join(inference_path, img_name)
            
            # Only convert if we haven't done it yet
            if not os.path.exists(img_path):
                try:
                    mel = torch.load(os.path.join(inference_path, file), map_location='cpu')
                    # Mel shape is [1, 80, T], squeeze to [80, T]
                    mel = mel.squeeze(0).detach().numpy()
                    
                    plt.figure(figsize=(10, 4))
                    plt.imshow(mel, aspect='auto', origin='lower', cmap='viridis')
                    plt.title(f"Mel Spectrogram: {file}")
                    plt.colorbar(format='%+2.0f dB')
                    plt.savefig(img_path)
                    plt.close()
                    print(f"🎨 Created visualization for {file}")
                except Exception as e:
                    print(f"❌ Failed to visualize {file}: {e}")

if __name__ == "__main__":
    print("🚀 Tacotron 2 Local Monitor Active.")
    print("Close this terminal to stop syncing.")
    
    try:
        while True:
            sync_from_cloud()
            convert_mels_to_images()
            print("✨ Progress updated. Next sync in 15 minutes...")
            time.sleep(900) # 15 minute wait
    except KeyboardInterrupt:
        print("\n🛑 Monitor stopped.")