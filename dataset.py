import os
import json
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class LibriTTSDataset(Dataset):
    def __init__(self, data_dir: str, validate_files: bool = False):
        """
        Args:
            data_dir:       Directory containing melspectrograms + metadata.json
            validate_files: If True, skip missing .npy files at init time (slower
                            startup but avoids surprises during training).
        """
        self.data_dir = data_dir
        self.file_paths: list[str] = []
        self.speaker_ids: list[int] = []
        self.speaker_to_int: dict[str, int] = {}

        metadata_path = os.path.join(data_dir, "metadata.json")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(f"metadata.json not found in {data_dir}")

        with open(metadata_path, "r") as f:
            metadata: dict[str, int] = json.load(f)

        logger.info("Mapping dataset paths...")

        for idx, (speaker_id, chunk_count) in enumerate(metadata.items()):
            self.speaker_to_int[speaker_id] = idx

            for i in range(chunk_count):
                path = os.path.join(data_dir, speaker_id, f"{speaker_id}_chunk_{i}.npy")
                if validate_files and not os.path.exists(path):
                    logger.warning(f"Missing file, skipping: {path}")
                    continue
                self.file_paths.append(path)
                self.speaker_ids.append(idx)

        logger.info(
            f"Dataset ready: {len(self.file_paths)} chunks across {len(metadata)} speakers."
        )

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.file_paths[idx]

        try:
            mel_spec = np.load(path)
        except Exception as e:
            raise RuntimeError(f"Failed to load {path}: {e}") from e

        # librosa saves as (n_mels, frames) → transpose to (frames, n_mels) for LSTM
        mel_tensor = torch.from_numpy(mel_spec).float().T

        speaker_label = torch.tensor(self.speaker_ids[idx], dtype=torch.long)

        return mel_tensor, speaker_label

    @property
    def num_speakers(self) -> int:
        """Convenience property — useful when constructing classifier heads."""
        return len(self.speaker_to_int)