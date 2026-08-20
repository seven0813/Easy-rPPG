"""LGI（Local Group Invariance）的可微 torch 实现。

对应 numpy 版本 ``methods/LGI.py``：

Pilz, C. S., Zaunseder, S., Krajewski, J. & Blazek, V. (2018). Local group
invariance for heart rate estimation from face videos. CVPRW, 1254-1262.

LGI 对逐帧 RGB 做 SVD，用第一左奇异向量构造投影 ``P = I - S S^T`` 把主
方向（光照/肤色）投影掉，取绿通道分量作为脉搏。``torch.linalg.svd`` 可微，
主奇异值通常与其余分离良好，反传稳定。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    spatial_rgb,
)


def lgi_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    apply_bandpass: bool = False,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
) -> Tensor:
    """LGI 投影后取绿通道作为 BVP，输出 ``[B,T]``。"""
    if rgb is None:
        if video is None:
            raise ValueError("either video or rgb must be provided")
        rgb = spatial_rgb(video, skin_mask)
    if rgb.ndim != 3 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,T]")

    # rgb: [B,3,T]；左奇异向量 U: [B,3,3]。
    left, _, _ = torch.linalg.svd(rgb, full_matrices=False)
    principal = left[:, :, 0:1]  # [B,3,1]
    projection = torch.eye(3, device=rgb.device, dtype=rgb.dtype)[None]
    projection = projection - principal @ principal.transpose(1, 2)
    projected = projection @ rgb  # [B,3,T]

    pulse = projected[:, 1]
    if apply_bandpass:
        pulse = bandpass_fft(pulse, fs, low_hz, high_hz)
    return pulse
