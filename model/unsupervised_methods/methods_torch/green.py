"""GREEN 方法的可微 torch 实现。

对应 numpy 版本 ``methods/GREEN.py``：

Verkruysse, W., Svaasand, L. O. & Nelson, J. S. (2008). Remote
plethysmographic imaging using ambient light. Optics Express, 16, 21434.

GREEN 直接取皮肤区域绿通道的逐帧空间平均作为脉搏信号，是最简单的基线。
numpy 版返回未滤波的原始绿通道均值；这里默认与其一致，并提供可选带通。
"""

from __future__ import annotations

from typing import Optional

from torch import Tensor

from model.unsupervised_methods.methods_torch._common import (
    bandpass_fft,
    spatial_rgb,
)


def green_torch(
    video: Optional[Tensor] = None,
    fs: float = 30.0,
    skin_mask: Optional[Tensor] = None,
    *,
    rgb: Optional[Tensor] = None,
    apply_bandpass: bool = False,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
) -> Tensor:
    """从视频（或预提取 RGB）取绿通道均值作为 BVP，输出 ``[B,T]``。

    ``apply_bandpass=False`` 时与 numpy ``GREEN`` 数学一致（返回原始绿通道
    均值）；作为可微回读监督时建议置 True，让信号落在生理频带。
    """
    if rgb is None:
        if video is None:
            raise ValueError("either video or rgb must be provided")
        rgb = spatial_rgb(video, skin_mask)
    if rgb.ndim != 3 or rgb.shape[1] != 3:
        raise ValueError("rgb must have shape [B,3,T]")

    pulse = rgb[:, 1]
    if apply_bandpass:
        pulse = bandpass_fft(pulse, fs, low_hz, high_hz)
    return pulse
