"""Datasets for reading cropped face clips and BVP labels from H5 files."""

import numpy as np
from .base_dataset import BaseH5ClipDataset




class H5Dataset(BaseH5ClipDataset):
    """Return one randomly sampled clip from each H5 file per access."""

    def __len__(self):
        return len(self.h5_infos)

    def __getitem__(self, index):
        info = self.h5_infos[index]
        max_start = info["length"] - self.clip_length
        start = np.random.randint(self.start_offset, max_start + 1)
        frames, label = self._read_clip(info, start)
        return frames, label, info["subject_id"], 0


class H5ClipOrderDataset(BaseH5ClipDataset):
    """Return all full clips from each H5 file in temporal order."""

    def __init__(self, config):
        super().__init__(config)
        self.stride = int(config.get("stride", self.clip_length))
        if self.stride <= 0:
            raise ValueError("stride must be greater than 0.")

        self.samples = []
        for h5_index, info in enumerate(self.h5_infos):
            max_start = info["length"] - self.clip_length
            for clip_index, start in enumerate(
                range(self.start_offset, max_start + 1, self.stride)
            ):
                self.samples.append((h5_index, start, clip_index))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        h5_index, start, clip_index = self.samples[index]
        info = self.h5_infos[h5_index]
        frames, label = self._read_clip(info, start)
        return frames, label, info["subject_id"], clip_index
