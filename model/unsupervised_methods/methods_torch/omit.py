"""OMIT（Orthogonal Matrix Image Transformation）的可微 torch 实现。

对应 numpy 版本 ``methods/OMIT.py``：

Álvarez Casado, C., & Bordallo López, M. (2023). Face2PPG: An unsupervised
pipeline for blood volume pulse extraction from faces. IEEE JBHI.

OMIT 对逐帧 RGB 做 QR 分解，用 Q 的第一列构造投影 ``P = I - S S^T``，投掉
主方向后取绿通道分量。``torch.linalg.qr`` 可微。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    spatial_rgb,
)


def omit_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    apply_bandpass: bool = False,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
) -> Tensor:
    """OMIT 投影后取绿通道作为 BVP，输出 ``[B,T]``。"""
    if rgb is None:
        if video is None:
            raise ValueError("either video or rgb must be provided")
        rgb = spatial_rgb(video, skin_mask)
    if rgb.ndim != 3 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,T]")

    # rgb: [B,3,T]；reduced QR 的 Q: [B,3,3]。
    orthogonal, _ = torch.linalg.qr(rgb, mode="reduced")
    principal = orthogonal[:, :, 0:1]  # [B,3,1]
    projection = torch.eye(3, device=rgb.device, dtype=rgb.dtype)[None]
    projection = projection - principal @ principal.transpose(1, 2)
    projected = projection @ rgb  # [B,3,T]

    pulse = projected[:, 1]
    if apply_bandpass:
        pulse = bandpass_fft(pulse, fs, low_hz, high_hz)
    return pulse
