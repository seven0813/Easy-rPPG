"""CHROM（Chrominance）方法的可微 torch 实现。

对应 numpy 版本 ``methods/CHROME_DEHAAN.py``：

De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based
rPPG. IEEE TBME, 60(10), 2878-2886.

保持与 numpy 版相同的分窗、色度投影、std 比例和 Hann 加窗重叠相加，只把
逐窗 ``filtfilt`` 换成可微的 FFT 带通，并向量化到 batch，可用于训练损失。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    spatial_rgb,
)


def chrom_dehaan_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    win_sec: float = 1.6,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    eps: float = 1e-8,
) -> Tensor:
    """从视频（或预提取的 RGB）可微地估计 CHROM BVP 波形。

    Args:
        video: 输入视频 ``[B,3,T,H,W]``，RGB、范围约 ``[0,1]``。
        fs: 采样率（帧率）。
        skin_mask: 可选皮肤 mask ``[B,1,T,H,W]``。
        rgb: 可选，已提取的逐帧空间平均 RGB ``[B,3,T]``；给定时忽略 video。
        win_sec: 分窗长度（秒），默认 1.6，内部向上取偶。
        low_hz, high_hz: 带通频带，默认 0.7-2.5 Hz。

    Returns:
        BVP 波形 ``[B,T]``（未做幅度归一化，交由损失侧标准化）。
    """
    if rgb is None:
        if video is None:
            raise ValueError("either video or rgb must be provided")
        rgb = spatial_rgb(video, skin_mask)
    if rgb.ndim != 3 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,T]")

    batch, _, length = rgb.shape
    window = math.ceil(win_sec * fs)
    if window % 2:
        window += 1
    if length < window:
        raise ValueError(
            f"sequence length {length} must be >= CHROM window {window}"
        )
    half = window // 2

    # 50% 重叠分窗：[B,3,num_windows,window]。
    windows = rgb.unfold(2, window, half)
    num_windows = windows.shape[2]
    base = windows.mean(dim=3, keepdim=True).clamp_min(eps)
    normalized = windows / base

    red = normalized[:, 0]
    green = normalized[:, 1]
    blue = normalized[:, 2]
    # CHROM 色度信号 Xs、Ys。
    x_signal = 3.0 * red - 2.0 * green
    y_signal = 1.5 * red + green - 1.5 * blue
    x_filtered = bandpass_fft(x_signal, fs, low_hz, high_hz)
    y_filtered = bandpass_fft(y_signal, fs, low_hz, high_hz)

    std_x = x_filtered.std(dim=2, unbiased=False, keepdim=True)
    std_y = y_filtered.std(dim=2, unbiased=False, keepdim=True).clamp_min(eps)
    alpha = std_x / std_y
    window_signal = x_filtered - alpha * y_filtered

    hann = torch.hann_window(
        window, periodic=False, device=rgb.device, dtype=rgb.dtype
    )
    window_signal = window_signal * hann

    # Hann 加窗重叠相加，跳步为半窗。
    output_length = half * (num_windows - 1) + window
    pulse = rgb.new_zeros(batch, output_length)
    for i in range(num_windows):
        start = i * half
        index = torch.arange(start, start + window, device=rgb.device)
        pulse = pulse.index_add(1, index, window_signal[:, i, :])

    if pulse.shape[1] > length:
        pulse = pulse[:, :length]
    elif pulse.shape[1] < length:
        pulse = F.pad(pulse, (0, length - pulse.shape[1]))
    return pulse
