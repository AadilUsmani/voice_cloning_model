# Voice Cloning Project - Copilot Instructions

This is a multi-stage voice cloning system combining speaker verification (encoder) and speech synthesis (Tacotron2), designed to run on Modal Labs infrastructure.

## Project Architecture

### Pipeline Overview

The system follows a 5-phase training pipeline:

1. **Data Ingestion** (`data_ingestion.py`)
   - Downloads LibriTTS train-clean-100 dataset (~25GB)
   - Extracts to Modal volume at `/data/LibriTTS/train-clean-100`

2. **Preprocessing** (`preprocess_data.py`)
   - Converts WAV files to 40-channel mel-spectrograms (encoder input)
   - Chunks audio into 1.6s segments
   - Outputs to `/data/melspectrograms/{speaker_id}/{speaker_id}_chunk_{i}.npy`

3. **Speaker Encoder Training** (`train.py`)
   - LSTM-based speaker verification model (`model.py`)
   - Uses metric learning (contrastive loss) via `pytorch-metric-learning`
   - Sharded data loading via `packer.py` (10 shards, float16 compression)
   - Saves checkpoints to `/data/checkpoints/encoder_best.pth`

4. **Synthesizer Dataset Preparation** (`prep_synthesize_data.py`)
   - Creates dual-pipeline mel-spectrograms:
     - 80-channel @ 22.05kHz (Tacotron2 input)
     - 40-channel @ 16kHz (encoder input for embeddings)
   - Generates `train.txt` with format: `{mel_path}|{speaker_embedding_path}|{text}`
   - Split into train/val sets via `split_data.py`

5. **Tacotron2 Training** (`train_tacotron2.py`)
   - Sequence-to-sequence TTS with location-sensitive attention
   - Speaker conditioning via concatenated embeddings (512-dim total)
   - Gradient accumulation for effective batch size on T4 GPU
   - Outputs checkpoints to `/data/tacotron2_checkpoints/`
   - Generates mel-spectrograms saved as `.pt` files during inference

6. **Vocoder (Mel-to-Audio Conversion)**
   - Tacotron2 produces 80-channel mel-spectrograms (`.pt` format)
   - Convert mels to audio using external vocoders (HiFi-GAN or WaveGlow)
   - Mel outputs: `/data/tacotron2_inference_samples/step_{N}_test_{idx}.pt`
   - See `training_progress/tacotron2_inference_samples/pt_file_test.py` for mel format verification

### Key Components

**Models:**
- `model.py`: SpeakerEncoder (LSTM → embedding projection → classifier)
- `tacotron2_model.py`: TextEncoder, Decoder, PostNet, LocationSensitiveAttention

**Datasets:**
- `dataset.py`: LibriTTSDataset (lazy-loading speaker encoder data)
- `synthesizer_dataset.py`: SynthesizerDataset (text + mel + speaker embeddings)

**Utilities:**
- `packer.py`: Shards mel-spectrograms into 10 `.pt` files for efficient loading
- `visualize.py`, `visualize_tacotron_attn.py`: Training progress visualization
- `check_data.py`, `verify_tensors.py`: Dataset validation tools

## Running on Modal

### Prerequisites

All scripts use Modal Labs for distributed compute. Ensure:
- Modal CLI installed: `pip install modal`
- Authenticated: `modal token new`
- Volume exists: `libritts-volume` (auto-created by `data_ingestion.py`)

### Deployment Commands

**Phase 1: Download dataset**
```bash
modal run data_ingestion.py
```

**Phase 2: Preprocess audio**
```bash
modal run preprocess_data.py
```

**Phase 3: Train speaker encoder**
```bash
modal run train.py
```

**Phase 4a: Prepare synthesizer dataset**
```bash
modal run prep_synthesize_data.py
```

**Phase 4b: Split train/val data**
```bash
modal run split_data.py
```

**Phase 5: Train Tacotron2**
```bash
modal run train_tacotron2.py
```

**Phase 6: Convert mel-spectrograms to audio**
- Inference mels are saved to `/data/tacotron2_inference_samples/`
- Use external vocoder (HiFi-GAN or WaveGlow) to convert `.pt` mels to `.wav`
- Format: `[n_mels, time_steps]` tensor (80 mel bins)

### Monitoring Training

Checkpoints and visualizations are saved to the Modal volume:
- Encoder: `/data/checkpoints/encoder_*.pth`
- Tacotron2: `/data/tacotron2_checkpoints/{latest|best}_checkpoint.pt`
- Attention plots: `/data/tacotron2_attention_plots/attention_step_{N}.png`
- Inference samples: `/data/tacotron2_inference_samples/step_{N}_test_{idx}.pt`

Use `monitor_progress.py` to download artifacts locally:
```bash
modal run monitor_progress.py  # Downloads from Modal volume to ./training_progress/
```

## Code Conventions

### Modal App Patterns

**Standard structure:**
```python
app = modal.App("app-name")
volume = modal.Volume.from_name("libritts-volume")
image = modal.Image.debian_slim().pip_install("torch", "numpy", ...)

@app.function(
    image=image,
    volumes={"/data": volume},
    gpu="T4",  # or "any" for CPU tasks
    timeout=3600
)
def training_function():
    # Training logic
    volume.commit()  # CRITICAL: Persist changes

@app.local_entrypoint()
def main():
    training_function.remote()
```

**Volume commit patterns:**
- Commit after best model updates (immediate persistence)
- Commit in `finally` blocks (crash safety)
- Commit every N processed items during data prep (`COMMIT_EVERY`)

### GPU Memory Management

**Critical for T4 (16GB VRAM):**
```python
# ALWAYS set at script top for Tacotron2 training
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
```

**Memory-efficient patterns:**
- Gradient accumulation (`GRADIENT_ACCUMULATION_STEPS = 8`)
- Small batch sizes with accumulation (effective batch = batch × accum_steps)
- `MAX_MEL_FRAMES` caps to avoid OOM on long utterances
- Float16 storage in data shards (loaded as float32)

### Dataset Loading

**Sharded loading pattern:**
```python
# packer.py creates shards:
for i in range(num_shards):
    shard = {"data": [(tensor, label), ...]}
    torch.save(shard, f"/data/dataset_shard_{i}.pt")

# train.py loads shards into RAM:
for i in range(NUM_SHARDS):
    shard = torch.load(path, weights_only=True)  # Security: prevent arbitrary code exec
    for tensor, label in shard["data"]:
        self.data_cache.append(tensor.float())  # float16 → float32
```

**Lazy loading pattern (synthesizer):**
```python
# Dataset stores only metadata (paths, text)
def __getitem__(self, idx):
    mel = np.load(self.data[idx]["mel_path"])  # Load on-demand
    embed = np.load(self.data[idx]["embed_path"])
    # ... process and return
```

### Text Processing

**Character vocabulary (synthesizer_dataset.py):**
```python
CHARACTERS = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?-'\'':;\"()"
VOCAB = ["<PAD>", "<EOS>"] + list(CHARACTERS)  # PAD=0, EOS=1
```

**Conversion:**
```python
def text_to_sequence(text: str) -> list[int]:
    seq = [CHAR_TO_IDX.get(c, CHAR_TO_IDX[" "]) for c in text]  # Fallback to space
    seq.append(EOS_IDX)  # Attention learns stop condition
    return seq
```

### Hyperparameter Tuning Guides

**Speaker Encoder (`train.py`):**
- `M_PER_CLASS = 4`: Samples per speaker in each batch (for contrastive loss)
- `BATCH_SIZE = 64`: Total samples per batch (must be divisible by M_PER_CLASS)
- `GRAD_CLIP = 5.0`: Gradient clipping threshold

**Tacotron2 (`train_tacotron2.py`):**
- `BATCH_SIZE = 8`: Physical batch size (limited by VRAM)
- `GRADIENT_ACCUMULATION_STEPS = 8`: Effective batch = 8 × 8 = 64
- `MAX_MEL_FRAMES = 600`: Max sequence length (reduce if OOM)
- `TF_DECAY_START = 10_000`: Step to begin teacher forcing decay
- `TF_DECAY_STEPS = 50_000`: Steps to decay from 1.0 → 0.5
- `TF_MIN = 0.5`: Minimum teacher forcing ratio

### Checkpoint Format

**Speaker encoder (torch.save state_dict):**
```python
state_dict = torch.load("encoder_best.pth")
state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}  # torch.compile cleanup
state_dict.pop("classifier.weight", None)  # Remove if fine-tuning without classification
model.load_state_dict(state_dict, strict=False)
```

**Tacotron2 (full checkpoint):**
```python
checkpoint = {
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "scaler_state_dict": scaler.state_dict(),  # AMP scaler
    "global_step": step,
    "epoch": epoch,
    "loss": loss
}
torch.save(checkpoint, path)
```

## Debugging Tips

### Common Issues

**OOM during Tacotron2 training:**
1. Reduce `BATCH_SIZE` (currently 8)
2. Increase `GRADIENT_ACCUMULATION_STEPS` to maintain effective batch size
3. Lower `MAX_MEL_FRAMES` (600 → 400)
4. Check dataset filtering: `synthesizer_dataset.py` skips samples > `MAX_MEL_FRAMES`

**Attention not converging:**
- Check teacher forcing schedule (`TF_DECAY_START`, `TF_DECAY_STEPS`)
- Visualize alignments: `visualize_tacotron_attn.py` on saved attention plots
- Ensure text padding uses `PAD_IDX = 0`
- Verify mask generation in `get_mask_from_lengths()`

**Volume persistence issues:**
- Always call `volume.commit()` after writes
- Wrap commits in try/finally for crash safety
- Check Modal logs: `modal app logs {app-name}`

### Validation Scripts

**Before training:**
```bash
python check_data.py          # Verify melspectrogram shapes
python verify_tensors.py      # Check packed shards
```

**During training:**
```bash
modal run monitor_progress.py  # Download checkpoints/plots
python visualize_tacotron_attn.py  # Analyze attention alignments
```

**After inference:**
```bash
python training_progress/tacotron2_inference_samples/pt_file_test.py  # Verify mel format
```

## File Naming Conventions

**Data paths:**
- Raw audio: `/data/LibriTTS/train-clean-100/{speaker_id}/{chapter_id}/{speaker_id}_{chapter_id}_{utterance_id}.wav`
- Mel-spectrograms: `/data/melspectrograms/{speaker_id}/{speaker_id}_chunk_{i}.npy`
- Synthesizer mels: `/data/synthesizer_dataset/mels/{speaker_id}/{utterance_id}.npy`
- Speaker embeddings: `/data/synthesizer_dataset/embeds/{speaker_id}/{utterance_id}.npy`

**Checkpoints:**
- Pattern: `{model}_epoch_{N}.pth` or `{model}_{best|final}.pth`
- Always include epoch/step numbers for resumption

**Attention plots:**
- `attention_step_{N}.png` (training progress)
- `step_{N}_test_{idx}_attention.png` (inference samples)

**Inference outputs:**
- Mel-spectrograms: `step_{N}_test_{idx}.pt` (80 mels × time_steps)
- Attention: `step_{N}_test_{idx}_attention.png`

## Project-Specific Notes

- **No local GPU required**: All training runs on Modal'\''s cloud GPUs
- **Dataset is immutable**: After preprocessing, data is read-only (safe for parallel access)
- **Two mel pipelines**: Encoder (40-channel @ 16kHz) vs. Synthesizer (80-channel @ 22.05kHz)
- **Speaker conditioning**: Embeddings concatenated to encoded text (not added), creating 512-dim input to decoder
- **Inference workflow**: Text → TextEncoder → concat(text_encoding, speaker_embedding) → Decoder → mel-spectrogram → Vocoder → audio
- **Vocoder not included**: Use external vocoder (HiFi-GAN or WaveGlow) to convert Tacotron2 mels to waveforms
