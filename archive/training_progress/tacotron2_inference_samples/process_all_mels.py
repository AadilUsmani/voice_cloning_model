import torch
import numpy as np
import os
import glob

script_dir = os.path.dirname(os.path.abspath(__file__))

# Process all .pt files
pt_files = glob.glob(os.path.join(script_dir, "step_*_test_*.pt"))

print(f"Found {len(pt_files)} mel-spectrogram files to process\n")

for pt_file in pt_files:
    base_name = os.path.basename(pt_file).replace('.pt', '')
    print(f"Processing: {base_name}")
    
    try:
        # Load mel
        mel_tensor = torch.load(pt_file, weights_only=True)
        
        # Remove batch dimension if present
        if len(mel_tensor.shape) == 3:
            mel_tensor = mel_tensor.squeeze(0)
        
        n_mels, time_steps = mel_tensor.shape
        
        # Save as numpy
        output_npy = os.path.join(script_dir, f"{base_name}_mel.npy")
        np.save(output_npy, mel_tensor.cpu().numpy())
        
        # Create visualization
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            
            plt.figure(figsize=(12, 4))
            plt.imshow(mel_tensor.cpu().numpy(), aspect='auto', origin='lower', cmap='viridis')
            plt.colorbar(label='Mel amplitude')
            plt.xlabel('Time frames')
            plt.ylabel('Mel bins')
            plt.title(f'Mel-Spectrogram: {base_name}')
            plt.tight_layout()
            
            output_png = os.path.join(script_dir, f"{base_name}_visualization.png")
            plt.savefig(output_png, dpi=150)
            plt.close()
            
            print(f"  ✅ Shape: {mel_tensor.shape} | Range: [{mel_tensor.min():.2f}, {mel_tensor.max():.2f}]")
            print(f"  📊 Saved: {output_npy}")
            print(f"  🖼️  Saved: {output_png}\n")
        except Exception as e:
            print(f"  ⚠️  Visualization failed: {e}\n")
            
    except Exception as e:
        print(f"  ❌ Error: {e}\n")

print("="*60)
print("All files processed!")
print(f"Output directory: {script_dir}")
print("="*60)
