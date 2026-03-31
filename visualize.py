import os
import logging
import torch
import modal
import numpy as np

logger = logging.getLogger(__name__)

# --- Modal Setup ---
app = modal.App("fyp-visualizer")
volume = modal.Volume.from_name("libritts-volume")

image = modal.Image.debian_slim().pip_install(
    "torch", "numpy", "matplotlib", "scikit-learn", "seaborn"
).add_local_file("model.py", remote_path="/root/model.py")

# --- Config ---
NUM_SPEAKERS = 10
MAX_SAMPLES = 500
SHARD_PATH = "/data/dataset_shard_0.pt"
CHECKPOINT_PATH = "/data/checkpoints/encoder_best.pth"
OUTPUT_DIR = "/data/visualizations"


def load_model_and_data(device: torch.device) -> tuple[object, np.ndarray, list[int]]:
    """Load model, run inference on a subset of speakers, return embeddings + labels."""
    from model import SpeakerEncoder

    model = SpeakerEncoder().to(device)
    
    # 1. Load the raw dictionary
    raw_state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    
    # 2. Strip the '_orig_mod.' prefix from the keys
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    
    # 3. Load the cleaned dictionary
    model.load_state_dict(clean_state_dict)
    model.eval()
    logger.info("Model loaded successfully after key cleaning.")

    shard = torch.load(SHARD_PATH, map_location=device, weights_only=True)
    target_speakers = set(list(shard["mapping"].values())[:NUM_SPEAKERS])

    embeddings, labels = [], []
    with torch.no_grad():
        for tensor, label in shard["data"]:
            if label not in target_speakers:
                continue
            emb = model(tensor.unsqueeze(0).float())
            embeddings.append(emb.squeeze(0).numpy())
            labels.append(label)
            if len(labels) >= MAX_SAMPLES:
                break

    logger.info(f"Generated {len(embeddings)} embeddings across {len(target_speakers)} speakers.")
    return model, np.array(embeddings), labels


def plot_tsne(embeddings: np.ndarray, labels: list[int], ax, palette):
    """t-SNE scatter — shows global cluster structure."""
    import seaborn as sns
    from sklearn.manifold import TSNE

    logger.info("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=labels, palette=palette,
        legend="full", alpha=0.75, s=60, ax=ax
    )
    ax.set_title("t-SNE: Speaker Embedding Clusters", fontsize=13)
    ax.set_xlabel("Dim 1")
    ax.set_ylabel("Dim 2")
    ax.legend(title="Speaker", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)


def plot_pca(embeddings: np.ndarray, labels: list[int], ax, palette):
    """PCA scatter — linear projection, fast, shows gross separation."""
    import seaborn as sns
    from sklearn.decomposition import PCA

    logger.info("Running PCA...")
    coords = PCA(n_components=2).fit_transform(embeddings)
    sns.scatterplot(
        x=coords[:, 0], y=coords[:, 1],
        hue=labels, palette=palette,
        legend=False, alpha=0.75, s=60, ax=ax
    )
    ax.set_title("PCA: Speaker Embedding Projection", fontsize=13)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")


def plot_cosine_heatmap(embeddings: np.ndarray, labels: list[int], ax):
    """
    Mean cosine similarity between every speaker pair.
    Diagonal should be high (intra-speaker), off-diagonal low (inter-speaker).
    A well-trained encoder shows a near-identity block structure.
    """
    import seaborn as sns
    from sklearn.preprocessing import normalize

    unique_speakers = sorted(set(labels))
    labels_arr = np.array(labels)
    normed = normalize(embeddings, norm="l2")

    # Compute mean centroid per speaker
    centroids = np.stack([
        normed[labels_arr == spk].mean(axis=0) for spk in unique_speakers
    ])
    centroids = normalize(centroids, norm="l2")
    sim_matrix = centroids @ centroids.T

    sns.heatmap(
        sim_matrix,
        annot=True, fmt=".2f",
        xticklabels=unique_speakers,
        yticklabels=unique_speakers,
        cmap="coolwarm", center=0,
        vmin=-1, vmax=1,
        ax=ax, annot_kws={"size": 7}
    )
    ax.set_title("Mean Cosine Similarity Between Speaker Centroids", fontsize=13)
    ax.set_xlabel("Speaker")
    ax.set_ylabel("Speaker")


def plot_distance_distributions(embeddings: np.ndarray, labels: list[int], ax):
    """
    Intra-speaker vs inter-speaker L2 distance distributions.
    Good separation = small intra overlap with large inter distances.
    """
    import seaborn as sns
    from itertools import combinations

    labels_arr = np.array(labels)
    unique_speakers = list(set(labels))

    intra, inter = [], []

    for spk in unique_speakers:
        spk_embs = embeddings[labels_arr == spk]
        for i, j in combinations(range(len(spk_embs)), 2):
            intra.append(np.linalg.norm(spk_embs[i] - spk_embs[j]))

    for s1, s2 in combinations(unique_speakers, 2):
        e1 = embeddings[labels_arr == s1]
        e2 = embeddings[labels_arr == s2]
        # Sample a bounded number of cross-speaker pairs to keep it fast
        for i in range(min(30, len(e1))):
            for j in range(min(30, len(e2))):
                inter.append(np.linalg.norm(e1[i] - e2[j]))

    sns.kdeplot(intra, ax=ax, label="Intra-speaker", fill=True, alpha=0.4, color="green")
    sns.kdeplot(inter, ax=ax, label="Inter-speaker", fill=True, alpha=0.4, color="red")
    ax.set_title("Intra vs Inter-Speaker L2 Distance Distribution", fontsize=13)
    ax.set_xlabel("L2 Distance")
    ax.set_ylabel("Density")
    ax.legend()


def plot_embedding_norms(embeddings: np.ndarray, ax):
    """
    L2 norm of each embedding (should be ~1.0 after normalization).
    Deviations hint at numerical instability or a broken normalization layer.
    """
    import seaborn as sns

    norms = np.linalg.norm(embeddings, axis=1)
    sns.histplot(norms, bins=30, kde=True, ax=ax, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", label="Expected norm = 1.0")
    ax.set_title("Embedding L2 Norm Distribution", fontsize=13)
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend()


@app.function(image=image, volumes={"/data": volume}, timeout=1200)
def plot_embeddings():
    import matplotlib.pyplot as plt
    import seaborn as sns

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    _, embeddings, labels = load_model_and_data(device)

    palette = sns.color_palette("hsv", NUM_SPEAKERS)

    # --- 2x2 overview figure ---
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    fig.suptitle("Speaker Encoder — Embedding Space Diagnostics", fontsize=16, y=1.01)

    plot_tsne(embeddings, labels, axes[0, 0], palette)
    plot_pca(embeddings, labels, axes[0, 1], palette)
    plot_distance_distributions(embeddings, labels, axes[1, 0])
    plot_embedding_norms(embeddings, axes[1, 1])

    plt.tight_layout()
    overview_path = os.path.join(OUTPUT_DIR, "overview.png")
    fig.savefig(overview_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Overview saved → {overview_path}")

    # --- Cosine similarity heatmap (separate — needs its own figure size) ---
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    plot_cosine_heatmap(embeddings, labels, ax2)
    plt.tight_layout()
    heatmap_path = os.path.join(OUTPUT_DIR, "cosine_similarity.png")
    fig2.savefig(heatmap_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    logger.info(f"Heatmap saved → {heatmap_path}")

    volume.commit()
    logger.info("All plots committed to volume.")


@app.local_entrypoint()
def main():
    plot_embeddings.remote()

if __name__ == "__main__":
    main()
