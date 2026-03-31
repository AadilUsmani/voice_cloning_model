import torch
import torch.nn as nn
import torch.nn.functional as F


class SpeakerEncoder(nn.Module):
    def __init__(
        self,
        mel_n_channels: int = 40,
        hidden_size: int = 256,
        num_layers: int = 3,
        embedding_size: int = 256,
        dropout: float = 0.1,
        n_speakers: int = 1000,
    ):
        super().__init__()

        # LSTM with built-in dropout between layers (only applied when num_layers > 1)
        self.lstm = nn.LSTM(
            input_size=mel_n_channels,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Layer norm stabilizes the LSTM output distribution across varying utterance lengths
        self.layer_norm = nn.LayerNorm(hidden_size)

        # Projection head: Linear → ReLU → Dropout → Linear
        # The extra layer gives the encoder more capacity to learn a clean embedding space
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, embedding_size),
        )

        # Classification head for auxiliary supervision
        self.classifier = nn.Linear(embedding_size, n_speakers)

    def forward(self, mels: torch.Tensor):
        # mels: (batch_size, frames, mel_channels)

        out, _ = self.lstm(mels)

        # Take the last frame — it has attended over the full sequence
        out = out[:, -1, :]

        # Normalize LSTM output before projection
        out = self.layer_norm(out)

        embeds = self.projection(out)

        # L2 normalize — required for Triplet / GE2E loss geometry
        # F.normalize handles near-zero vectors safely (eps guard built in)
        embeddings = F.normalize(embeds, p=2, dim=1)

        # During training, return both embeddings and logits for auxiliary classification loss
        if self.training:
            logits = self.classifier(embeddings)
            return embeddings, logits
        
        # During inference, only return embeddings
        return embeddings




