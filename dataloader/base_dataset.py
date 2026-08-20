import os
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


_TYPE_ALIASES = {
    "raw": "Raw",
    "standardized": "Standardized",
    "diffnormalized": "DiffNormalized",
}


def canonicalize_signal_type(value, field_name):
    """Return the canonical spelling of a data/label representation type."""
    if not isinstance(value, str):
        raise TypeError(
            f"{field_name} must be a string, got {type(value).__name__}."
        )
    normalized = value.strip().replace("_", "").replace("-", "").lower()
    canonical = _TYPE_ALIASES.get(normalized)
    if canonical is None:
        supported = ", ".join(_TYPE_ALIASES.values())
        raise ValueError(
            f"Unsupported {field_name} {value!r}. Supported: {supported}."
        )
    return canonical


def canonicalize_data_types(value):
    """Normalize a DATA_TYPE-style string/list while preserving channel order."""
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError("data_type must be a non-empty string or list of strings.")
    return tuple(canonicalize_signal_type(item, "data_type") for item in value)


def _standardize(tensor):
    tensor = tensor - tensor.mean()
    std = tensor.std(unbiased=False)
    if not torch.isfinite(std) or std <= torch.finfo(tensor.dtype).eps:
        return torch.zeros_like(tensor)
    return tensor / std


def _diff_normalize_frames(frames):
    """Apply toolbox-style relative temporal differences to [T,C,H,W]."""
    if frames.shape[0] < 2:
        return torch.zeros_like(frames)
    current = frames[:-1]
    following = frames[1:]
    differences = (following - current) / (following + current + 1e-7)
    std = differences.std(unbiased=False)
    if not torch.isfinite(std) or std <= torch.finfo(frames.dtype).eps:
        differences = torch.zeros_like(differences)
    else:
        differences = differences / std
    return torch.cat((differences, torch.zeros_like(frames[:1])), dim=0)


def _diff_normalize_label(label):
    """Apply first differences/std and append zero to preserve label length."""
    if label.shape[0] < 2:
        return torch.zeros_like(label)
    differences = label[1:] - label[:-1]
    std = differences.std(unbiased=False)
    if not torch.isfinite(std) or std <= torch.finfo(label.dtype).eps:
        differences = torch.zeros_like(differences)
    else:
        differences = differences / std
    return torch.cat((differences, torch.zeros_like(label[:1])), dim=0)


class BaseH5ClipDataset(Dataset):
    """Shared H5 list parsing, validation, and clip reading."""

    def __init__(self, config):
        self.list_path = config.h5_path
        self.clip_length = config.clip_length
        self.start_offset = int(config.get("start_offset", 0))
        self.img_key = config.get("img_key", "imgs")
        self.label_key = config.get("label_key", "bvp")
        self.normalize = config.get("normalize", True)
        configured_data_type = config.get("data_type")
        self.data_types = (
            None
            if configured_data_type is None
            else canonicalize_data_types(configured_data_type)
        )
        self.label_type = canonicalize_signal_type(
            config.get("label_type", "Raw"), "label_type"
        )
        self.dataset = config.get("dataset", "unknown")
        self.resize = self._parse_resize(config.get("resize"))

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

    @staticmethod
    def _parse_resize(resize):
        """Parse YAML resize config and return (height, width), or None."""
        if resize is None:
            return None
        height = resize.get("height", resize.get("h"))
        width = resize.get("width", resize.get("w"))
        if height is None or width is None:
            raise ValueError("resize must define height/width or h/w.")
        height, width = int(height), int(width)
        if height <= 0 or width <= 0:
            raise ValueError("resize height and width must be greater than 0.")
        return height, width

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
        frames = frames.permute(0, 3, 1, 2).contiguous()
        if self.resize is not None and frames.shape[-2:] != self.resize:
            frames = F.interpolate(
                frames,
                size=self.resize,
                mode="bilinear",
                align_corners=False,
            )
        frames = self._transform_frames(frames)
        frames = frames.permute(1, 0, 2, 3).contiguous()
        label = torch.from_numpy(np.asarray(label, dtype=np.float32))
        label = self._transform_label(label)
        return frames, label

    def _transform_frames(self, frames):
        """Apply configured representations and concatenate them by channel."""
        if self.data_types is None:
            return frames

        transformed = []
        for data_type in self.data_types:
            if data_type == "Raw":
                transformed.append(frames.clone())
            elif data_type == "Standardized":
                transformed.append(_standardize(frames))
            elif data_type == "DiffNormalized":
                transformed.append(_diff_normalize_frames(frames))
        return torch.cat(transformed, dim=1)

    def _transform_label(self, label):
        if self.label_type == "Raw":
            return label
        if self.label_type == "Standardized":
            return _standardize(label)
        if self.label_type == "DiffNormalized":
            return _diff_normalize_label(label)
        raise AssertionError(f"Unhandled label_type: {self.label_type}")
