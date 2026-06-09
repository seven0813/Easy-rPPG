import os
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class BaseH5ClipDataset(Dataset):
    """Shared H5 list parsing, validation, and clip reading."""

    def __init__(self, config):
        self.list_path = config.h5_path
        self.clip_length = config.clip_length
        self.start_offset = int(config.get("start_offset", 0))
        self.img_key = config.get("img_key", "imgs")
        self.label_key = config.get("label_key", "bvp")
        self.normalize = config.get("normalize", True)
        self.dataset = config.get("dataset", "unknown")

        if self.clip_length <= 0:
            raise ValueError("clip_length must be greater than 0.")
        if self.start_offset < 0:
            raise ValueError("start_offset must be greater than or equal to 0.")

        self.h5_paths = self._read_h5_list(self.list_path)
        self.h5_infos = [self._inspect_h5(path) for path in self.h5_paths]

        invalid_paths = [
            info["path"]
            for info in self.h5_infos
            if info["length"] < self.start_offset + self.clip_length
        ]
        if invalid_paths:
            preview = ", ".join(invalid_paths[:3])
            raise ValueError(
                f"{len(invalid_paths)} H5 files are shorter than "
                f"start_offset + clip_length={self.start_offset + self.clip_length}. "
                f"Examples: {preview}"
            )

    @staticmethod
    def _read_h5_list(list_path):
        list_path = Path(list_path).expanduser().resolve()
        if not list_path.is_file():
            raise FileNotFoundError(f"H5 list file does not exist: {list_path}")

        paths = []
        with list_path.open("r", encoding="utf-8") as file:
            for line in file:
                path = line.strip()
                if not path or path.startswith("#"):
                    continue
                h5_path = Path(path).expanduser()
                if not h5_path.is_absolute():
                    h5_path = list_path.parent / h5_path
                paths.append(str(h5_path.resolve()))

        if not paths:
            raise ValueError(f"H5 list file is empty: {list_path}")
        return paths

    def _inspect_h5(self, h5_path):
        if not os.path.isfile(h5_path):
            raise FileNotFoundError(f"H5 file does not exist: {h5_path}")

        with h5py.File(h5_path, "r") as file:
            missing_keys = [
                key for key in (self.img_key, self.label_key) if key not in file
            ]
            if missing_keys:
                raise KeyError(f"{h5_path} is missing H5 keys: {missing_keys}")

            frame_length = file[self.img_key].shape[0]
            label_length = file[self.label_key].shape[0]
            length = min(frame_length, label_length)
            subject_id = self._split_id(h5_path)

        return {
            "path": h5_path,
            "length": length,
            "subject_id": subject_id,
        }
    def _split_id(self, h5_path):
        if self.dataset == "UBFC-rPPG":
            return os.path.basename(os.path.dirname(h5_path)) or os.path.basename(h5_path)
        if self.dataset in ("PURE", "BUAA"):
            return os.path.splitext(os.path.basename(h5_path))[0]
        return os.path.splitext(os.path.basename(h5_path))[0]

    def _read_clip(self, info, start):
        end = start + self.clip_length
        with h5py.File(info["path"], "r") as file:
            frames = file[self.img_key][start:end]
            label = file[self.label_key][start:end]

        frames = torch.from_numpy(np.asarray(frames, dtype=np.float32))
        if self.normalize:
            frames = frames / 255.0
        frames = frames.permute(3, 0, 1, 2).contiguous()
        label = torch.from_numpy(np.asarray(label, dtype=np.float32))
        return frames, label
