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

    # Load shard first to get actual speaker count
    shard = torch.load(SHARD_PATH, map_location=device, weights_only=True)
    n_speakers = len(shard.get("mapping", {}))
    logger.info(f"Detected {n_speakers} unique speakers in dataset")

    model = SpeakerEncoder(n_speakers=n_speakers).to(device)
    
    # 1. Load the raw dictionary
    raw_state_dict = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=True)
    
    # 2. Strip the '_orig_mod.' prefix from the keys
    clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in raw_state_dict.items()}
    
    # 3. Load the cleaned dictionary (strict=False to ignore the classifier head if needed)
    model.load_state_dict(clean_state_dict, strict=False)
    model.eval()
    logger.info("Model loaded successfully after key cleaning.")
    target_speakers = set(list(shard["mapping"].values())[:NUM_SPEAKERS])

    embeddings, labels = [], []
    with torch.no_grad():
        for tensor, label in shard["data"]:
            if label not in target_speakers:
                continue
            
            # Unpack the tuple since forward() now returns (embeddings, logits) during training
            # Note: since model.eval() is called, it might just return embeddings. We handle both cases.
            output = model(tensor.unsqueeze(0).float())
            emb = output[0] if isinstance(output, tuple) else output
            
            embeddings.append(emb.squeeze(0).numpy())
            labels.append(label)
            if len(labels) >= MAX_SAMPLES:
                break

    logger.info(f"Generated {len(embeddings)} embeddings across {len(target_speakers)} speakers.")
    return model, np.array(embeddings), labels


def calculate_and_log_metrics(embeddings: np.ndarray, labels: list[int]):
    """Calculates hard quantitative metrics to prove the clustering worked."""
    from sklearn.metrics import silhouette_score
    from sklearn.metrics.pairwise import cosine_similarity
    
    labels_arr = np.array(labels)
    
    logger.info("=== QUANTITATIVE VERIFICATION METRICS ===")
    
    # 1. Silhouette Score
    sil_score = silhouette_score(embeddings, labels_arr, metric='cosine')
    logger.info(f"📊 Silhouette Score: {sil_score:.4f} (Target: > 0.15 for deep embeddings. < 0 means mode collapse)")
    
    # 2. Cosine Similarities
    sim_matrix = cosine_similarity(embeddings)
    intra_sims, inter_sims = [], []
    
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if labels[i] == labels[j]:
                intra_sims.append(sim_matrix[i, j])
            else:
                inter_sims.append(sim_matrix[i, j])
                
    avg_intra = np.mean(intra_sims) if intra_sims else 0
    avg_inter = np.mean(inter_sims) if inter_sims else 0
    
    logger.info(f"🟢 Avg Intra-Speaker Similarity (Same person):     {avg_intra:.4f} (Target: Close to 1.0)")
    logger.info(f"🔴 Avg Inter-Speaker Similarity (Different people): {avg_inter:.4f} (Target: Close to 0.0)")
    
    # Automated Verdict
    margin = avg_intra - avg_inter
    logger.info(f"📐 Separation Margin: {margin:.4f}")
    
    if margin > 0.3:
        logger.info("✅ VERDICT: EXCELLENT PASS - Strong mathematical separation between voices!")
    elif margin > 0.1:
        logger.info("⚠️ VERDICT: MARGINAL PASS - Voices are separating, but clusters are a bit loose.")
    else:
        logger.info("❌ VERDICT: FAIL - The model cannot tell speakers apart (Mode Collapse).")
        
    logger.info("=========================================")


def plot_tsne(embeddings: np.ndarray, labels: list[int], ax, palette):
    import seaborn as sns
    from sklearn.manifold import TSNE
    logger.info("Running t-SNE...")
    coords = TSNE(n_components=2, perplexity=30, random_state=42).fit_transform(embeddings)
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette=palette, legend="full", alpha=0.75, s=60, ax=ax)
    ax.set_title("t-SNE: Speaker Embedding Clusters", fontsize=13)

def plot_pca(embeddings: np.ndarray, labels: list[int], ax, palette):
    import seaborn as sns
    from sklearn.decomposition import PCA
    logger.info("Running PCA...")
    coords = PCA(n_components=2).fit_transform(embeddings)
    sns.scatterplot(x=coords[:, 0], y=coords[:, 1], hue=labels, palette=palette, legend=False, alpha=0.75, s=60, ax=ax)
    ax.set_title("PCA: Speaker Embedding Projection", fontsize=13)

def plot_cosine_heatmap(embeddings: np.ndarray, labels: list[int], ax):
    import seaborn as sns
    from sklearn.preprocessing import normalize
    unique_speakers = sorted(set(labels))
    labels_arr = np.array(labels)
    normed = normalize(embeddings, norm="l2")
    centroids = np.stack([normed[labels_arr == spk].mean(axis=0) for spk in unique_speakers])
    centroids = normalize(centroids, norm="l2")
    sim_matrix = centroids @ centroids.T
    sns.heatmap(sim_matrix, annot=True, fmt=".2f", xticklabels=unique_speakers, yticklabels=unique_speakers, cmap="coolwarm", center=0, vmin=-1, vmax=1, ax=ax, annot_kws={"size": 7})
    ax.set_title("Mean Cosine Similarity Between Speaker Centroids", fontsize=13)

def plot_distance_distributions(embeddings: np.ndarray, labels: list[int], ax):
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
        for i in range(min(30, len(e1))):
            for j in range(min(30, len(e2))):
                inter.append(np.linalg.norm(e1[i] - e2[j]))
    sns.kdeplot(intra, ax=ax, label="Intra-speaker", fill=True, alpha=0.4, color="green")
    sns.kdeplot(inter, ax=ax, label="Inter-speaker", fill=True, alpha=0.4, color="red")
    ax.set_title("Intra vs Inter-Speaker L2 Distance Distribution", fontsize=13)
    ax.legend()

def plot_embedding_norms(embeddings: np.ndarray, ax):
    import seaborn as sns
    norms = np.linalg.norm(embeddings, axis=1)
    sns.histplot(norms, bins=30, kde=True, ax=ax, color="steelblue")
    ax.axvline(1.0, color="red", linestyle="--", label="Expected norm = 1.0")
    ax.set_title("Embedding L2 Norm Distribution", fontsize=13)
    ax.legend()


@app.function(image=image, volumes={"/data": volume}, timeout=1200)
def plot_embeddings():
    import matplotlib.pyplot as plt
    import seaborn as sns

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    device = torch.device("cpu")
    _, embeddings, labels = load_model_and_data(device)

    # ---> NEW METRICS LOGGING <---
    calculate_and_log_metrics(embeddings, labels)

    palette = sns.color_palette("hsv", len(set(labels)))

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
    