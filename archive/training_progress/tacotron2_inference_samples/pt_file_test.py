import torch
import numpy as np
import os

# 1. Handle the folder issue by getting the absolute path of the script
script_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(script_dir, "step_5000_test_0.pt")

print(f"Loading mel-spectrogram from: {file_path}")

# Load the tensor
mel_tensor = torch.load(file_path, weights_only=True)
print(f"Original tensor shape: {mel_tensor.shape}")

# 2. Analyze the mel-spectrogram
if len(mel_tensor.shape) == 3:
    # [batch, mels, time]
    mel_tensor = mel_tensor.squeeze(0)  # Remove batch dimension
    print(f"After squeeze: {mel_tensor.shape}")

n_mels, time_steps = mel_tensor.shape
print(f"Mel bins: {n_mels}, Time steps: {time_steps}")
print(f"Mel range: [{mel_tensor.min():.2f}, {mel_tensor.max():.2f}]")

# 3. Save as numpy for visualization or vocoder input
output_npy = os.path.join(script_dir, "test_0_mel.npy")
np.save(output_npy, mel_tensor.cpu().numpy())
print(f"✅ Mel-spectrogram saved to: {output_npy}")

# 4. Create a simple visualization
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 4))
    plt.imshow(mel_tensor.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
    plt.colorbar(label='Mel amplitude')
    plt.xlabel('Time frames')
    plt.ylabel('Mel bins')
    plt.title(f'Generated Mel-Spectrogram (Step 5000, Test 0)')
    plt.tight_layout()
    
    output_png = os.path.join(script_dir, "test_0_mel_visualization.png")
    plt.savefig(output_png, dpi=150)
    plt.close()
    print(f"✅ Visualization saved to: {output_png}")
except Exception as e:
    print(f"⚠️  Could not create visualization: {e}")

print("\n" + "="*60)
print("SUMMARY:")
print(f"  Input:  {file_path}")
print(f"  Output: {output_npy}")
print(f"  Shape:  {mel_tensor.shape} ({n_mels} mels × {time_steps} frames)")
print(f"  Range:  [{mel_tensor.min():.2f}, {mel_tensor.max():.2f}]")
print("\nTo convert to audio, you need a vocoder (HiFi-GAN or WaveGlow)")
print("="*60)