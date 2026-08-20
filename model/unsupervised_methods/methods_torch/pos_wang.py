"""POS（Plane-Orthogonal-to-Skin）的可微 torch 实现。

对应 numpy 版本 ``methods/POS_WANG.py``：

Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2017).
Algorithmic principles of remote PPG. IEEE TBME, 64(7), 1479-1491.

与 numpy 版的区别只在于把不可微算子替换为等价可微实现（``filtfilt``
换成 FFT 零相位带通、``detrend`` 换成预计算线性算子），并向量化到 batch。
因此该函数可以直接放进训练损失，让梯度从 BVP 回传到输入视频。
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    detrend,
    spatial_rgb,
)


def pos_wang_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    win_sec: float = 1.6,
    detrend_lambda: float = 100.0,
    low_hz: float = 0.75,
    high_hz: float = 3.0,
    eps: float = 1e-8,
) -> Tensor:
    """从视频（或预提取的 RGB）可微地估计 POS BVP 波形。

    Args:
        video: 输入视频 ``[B,3,T,H,W]``，RGB、范围约 ``[0,1]``。
        fs: 采样率（帧率）。
        skin_mask: 可选皮肤 mask ``[B,1,T,H,W]``。
        rgb: 可选，已提取的逐帧空间平均 RGB ``[B,3,T]``；给定时忽略 video。
        win_sec: POS 滑窗长度（秒），默认 1.6。
        detrend_lambda: 去趋势平滑强度，与 numpy 版一致默认 100。
        low_hz, high_hz: 带通频带，默认 0.75-3.0 Hz。

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
    if length <= window:
        raise ValueError(
            f"sequence length {length} must exceed POS window {window}"
        )
    num_windows = length - window

    # [B,3,num_windows,window]：与 numpy 的逐 n 滑窗（步长 1）对应。
    windows = rgb.unfold(2, window, 1)[:, :, :num_windows, :]
    # 每个窗口内按时间均值做时序归一化（temporal normalization）。
    base = windows.mean(dim=3, keepdim=True).clamp_min(eps)
    normalized = windows / base

    red = normalized[:, 0]
    green = normalized[:, 1]
    blue = normalized[:, 2]
    # 固定 POS 投影矩阵 [[0,1,-1],[-2,1,1]]。
    projection_0 = green - blue
    projection_1 = -2.0 * red + green + blue
    std_0 = projection_0.std(dim=2, unbiased=False, keepdim=True)
    std_1 = projection_1.std(dim=2, unbiased=False, keepdim=True).clamp_min(eps)
    window_signal = projection_0 + (std_0 / std_1) * projection_1
    window_signal = window_signal - window_signal.mean(dim=2, keepdim=True)

    # 重叠相加（步长 1）：窗口 w 的第 j 个采样贡献到时间 w+j。
    pulse = rgb.new_zeros(batch, length)
    for offset in range(window):
        index = torch.arange(
            offset, offset + num_windows, device=rgb.device
        )
        pulse = pulse.index_add(1, index, window_signal[:, :, offset])

    pulse = detrend(pulse, detrend_lambda)
    pulse = bandpass_fft(pulse, fs, low_hz, high_hz)
    return pulse
