import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def plot_attention(alignment, title="Attention Alignment", save_path=None):
    """
    Plot attention alignment heatmap
    
    Args:
        alignment: numpy array [decoder_steps, encoder_steps]
        title: Plot title
        save_path: Path to save the plot (if None, will display)
    
    Diagnostic patterns:
    - ✅ Diagonal line: Model learning proper text-to-audio alignment (GOOD)
    - ❌ Vertical lines: Attention stuck on single character (mode collapse)
    - ❌ Random noise: Model not converging (reduce learning rate)
    - ❌ Horizontal lines: Decoder ignoring text input (check embedding injection)
    """
    fig, ax = plt.subplots(figsize=(14, 8))
    
    im = ax.imshow(alignment, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
    
    fig.colorbar(im, ax=ax, label='Attention Weight')
    ax.set_xlabel('Encoder Timestep (Text Characters)', fontsize=12)
    ax.set_ylabel('Decoder Timestep (Mel Frames)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Attention plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def analyze_attention_quality(alignment):
    """
    Analyze attention alignment quality
    
    Returns:
        dict with diagnostic metrics
    """
    # Convert to numpy if tensor
    if torch.is_tensor(alignment):
        alignment = alignment.cpu().numpy()
    
    decoder_steps, encoder_steps = alignment.shape
    
    # 1. Monotonicity score (should increase along diagonal)
    # Compute weighted average encoder position for each decoder step
    encoder_positions = np.arange(encoder_steps)
    attention_progress = []
    for t in range(decoder_steps):
        weighted_pos = np.sum(alignment[t] * encoder_positions)
        attention_progress.append(weighted_pos)
    
    # Check if generally increasing
    monotonic_score = np.mean(np.diff(attention_progress) >= 0)
    
    # 2. Focus score (attention should be concentrated, not diffuse)
    entropy = -np.sum(alignment * np.log(alignment + 1e-8), axis=1)
    max_entropy = np.log(encoder_steps)
    normalized_entropy = entropy / max_entropy
    focus_score = 1.0 - np.mean(normalized_entropy)
    
    # 3. Coverage (should attend to most of the input)
    coverage = np.sum(alignment, axis=0)
    coverage_score = np.mean(coverage > 0.01)
    
    # 4. Diagonal alignment (should follow diagonal pattern)
    expected_positions = np.linspace(0, encoder_steps-1, decoder_steps)
    diagonal_deviation = np.abs(np.array(attention_progress) - expected_positions)
    diagonal_score = 1.0 - np.mean(diagonal_deviation) / encoder_steps
    
    return {
        'monotonic_score': float(monotonic_score),
        'focus_score': float(focus_score),
        'coverage_score': float(coverage_score),
        'diagonal_score': float(diagonal_score),
        'overall_quality': float((monotonic_score + focus_score + coverage_score + diagonal_score) / 4)
    }

def plot_multiple_attentions(alignments, titles, save_path=None):
    """
    Plot multiple attention alignments in a grid for comparison
    
    Args:
        alignments: list of numpy arrays
        titles: list of titles for each subplot
        save_path: Path to save the plot
    """
    n = len(alignments)
    rows = (n + 1) // 2
    cols = 2 if n > 1 else 1
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 6*rows))
    
    if n == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for i, (alignment, title) in enumerate(zip(alignments, titles)):
        im = axes[i].imshow(alignment, aspect='auto', origin='lower', interpolation='none', cmap='viridis')
        axes[i].set_xlabel('Encoder Timestep')
        axes[i].set_ylabel('Decoder Timestep')
        axes[i].set_title(title)
        axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        fig.colorbar(im, ax=axes[i])
    
    # Hide extra subplots
    for i in range(len(alignments), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✅ Comparison plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()

def visualize_checkpoint_attention(checkpoint_path, save_dir=None):
    """
    Load a checkpoint and visualize its attention patterns
    
    Args:
        checkpoint_path: Path to .pth checkpoint file
        save_dir: Directory to save plots (if None, will display)
    """
    print(f"📂 Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Extract step number
    step = checkpoint.get('step', 'unknown')
    
    print(f"✅ Loaded checkpoint at step {step}")
    print(f"   Loss: {checkpoint.get('loss', 'N/A')}")
    
    # Note: This is a utility function. Actual attention tensors need to be
    # saved separately during training, as checkpoints only contain model weights.
    print("\n⚠️  To visualize attention, you need to save attention tensors during training.")
    print("   The training script already does this in the INFERENCE_DIR.")

if __name__ == "__main__":
    # Example: Generate synthetic attention pattern for testing
    print("🧪 Testing attention visualization with synthetic data...\n")
    
    # Good diagonal alignment (what we want)
    decoder_steps = 200
    encoder_steps = 50
    good_attention = np.zeros((decoder_steps, encoder_steps))
    for t in range(decoder_steps):
        center = int((t / decoder_steps) * encoder_steps)
        for e in range(max(0, center-2), min(encoder_steps, center+3)):
            good_attention[t, e] = np.exp(-((e - center)**2) / 2)
    good_attention /= good_attention.sum(axis=1, keepdims=True)
    
    # Bad attention (stuck on one character)
    bad_attention = np.zeros((decoder_steps, encoder_steps))
    bad_attention[:, 10] = 1.0
    
    # Random attention (not converging)
    random_attention = np.random.rand(decoder_steps, encoder_steps)
    random_attention /= random_attention.sum(axis=1, keepdims=True)
    
    # Plot examples
    plot_attention(good_attention, "Good Alignment (Diagonal)", "test_good_attention.png")
    plot_attention(bad_attention, "Bad Alignment (Mode Collapse)", "test_bad_attention.png")
    plot_attention(random_attention, "Bad Alignment (Not Converging)", "test_random_attention.png")
    
    # Analyze quality
    print("\n📊 Attention Quality Analysis:")
    print("\n1. Good Alignment:")
    metrics = analyze_attention_quality(good_attention)
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n2. Bad Alignment (Mode Collapse):")
    metrics = analyze_attention_quality(bad_attention)
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    print("\n3. Random Alignment (Not Converging):")
    metrics = analyze_attention_quality(random_attention)
    for key, value in metrics.items():
        print(f"   {key}: {value:.3f}")
    
    # Comparison plot
    plot_multiple_attentions(
        [good_attention, bad_attention, random_attention],
        ["Good (Diagonal)", "Bad (Mode Collapse)", "Bad (Random)"],
        "test_attention_comparison.png"
    )
    
    print("\n✅ Test plots generated successfully!")
    print("   - test_good_attention.png")
    print("   - test_bad_attention.png")
    print("   - test_random_attention.png")
    print("   - test_attention_comparison.png")
