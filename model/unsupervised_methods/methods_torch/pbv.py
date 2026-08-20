"""PBV（Blood Volume Pulse signature）的可微 torch 实现。

对应 numpy 版本 ``methods/PBV.py``：

De Haan, G. & Van Leest, A. (2014). Improved motion robustness of remote-PPG
by using the blood volume pulse signature. Physiol. Meas., 35, 1913.

PBV 用血容脉搏的固定色度签名向量，通过求解 ``Q W = pbv`` 得到通道权重，
再把归一化 RGB 投影到该方向。``torch.linalg.solve`` 与 std/var 均可微。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    spatial_rgb,
)


def pbv_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    apply_bandpass: bool = False,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    eps: float = 1e-8,
) -> Tensor:
    """PBV 投影得到 BVP，输出 ``[B,T]``，与 numpy ``PBV`` 数学一致。"""
    if rgb is None:
        if video is None:
            raise ValueError("either video or rgb must be provided")
        rgb = spatial_rgb(video, skin_mask)
    if rgb.ndim != 3 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,T]")

    # 逐通道按时间均值归一化。
    channel_mean = rgb.mean(dim=2, keepdim=True).clamp_min(eps)  # [B,3,1]
    normalized = rgb / channel_mean  # [B,3,T]

    # 血容签名向量 pbv：各通道标准差 / 总方差平方根。
    channel_std = normalized.std(dim=2, unbiased=False)  # [B,3]
    total_std = channel_std.square().sum(dim=1, keepdim=True).clamp_min(eps).sqrt()
    pbv = channel_std / total_std  # [B,3]

    # Q = C C^T，W = Q^{-1} pbv。
    covariance = normalized @ normalized.transpose(1, 2)  # [B,3,3]
    weight = torch.linalg.solve(covariance, pbv.unsqueeze(-1))  # [B,3,1]

    numerator = normalized.transpose(1, 2) @ weight  # [B,T,1]
    denominator = (pbv.unsqueeze(1) @ weight).clamp_min(eps)  # [B,1,1]
    pulse = (numerator / denominator).squeeze(-1)  # [B,T]

    if apply_bandpass:
        pulse = bandpass_fft(pulse, fs, low_hz, high_hz)
    return pulse
