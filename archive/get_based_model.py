import urllib.request
import os

def show_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(downloaded * 100 / total_size, 100.0)
        print(f"\rDownloading Base Model: {percent:.1f}%", end="")

# Direct link to the raw weight file on NVIDIA's servers
url = "https://api.ngc.nvidia.com/v2/models/nvidia/tacotron2_pyt_ckpt_fp32/versions/19.09.0/files/nvidia_tacotron2pyt_fp32_20190427"
save_path = "tacotron2_ljspeech.pth"

print("⏳ Starting direct download (Bypassing PyTorch Hub)...")
try:
    urllib.request.urlretrieve(url, save_path, reporthook=show_progress)
    file_size = os.path.getsize(save_path) / (1024 * 1024)
    print(f"\n✅ Success! File saved: {save_path} ({file_size:.2f} MB)")
except Exception as e:
    print(f"\n❌ Download failed: {e}")