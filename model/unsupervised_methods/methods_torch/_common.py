"""可微 rPPG 传统方法的共享工具。

这些工具把 ``methods/`` 下 numpy/scipy 版本中不可微的算子替换为等价的
可微实现，供 ``methods_torch/`` 中的 POS、CHROM 等方法复用：

- :func:`spatial_rgb` 从视频（可选皮肤 mask）提取逐帧空间平均 RGB；
- :func:`bandpass_fft` 用 FFT 频域掩码实现零相位带通，替代 ``scipy.filtfilt``；
- :func:`detrend` 用预计算的固定线性算子实现 Tarvainen 平滑先验去趋势，
  替代 numpy 中每次调用都做矩阵求逆的实现。

所有函数都保持梯度可回传，并支持 ``[B, ...]`` 批量输入。
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
from torch import Tensor


def spatial_rgb(
    video: Tensor,
    skin_mask: Optional[Tensor] = None,
    eps: float = 1e-6,
) -> Tensor:
    """把 ``[B,3,T,H,W]`` 视频压成逐帧空间平均 RGB ``[B,3,T]``。

    传入 ``skin_mask``（``[B,1,T,H,W]``）时按皮肤区域面积做加权平均，
    以贴近传统方法在人脸/皮肤 ROI 上的取值方式。
    """
    if video.ndim != 5 or video.shape[1] != 3:
        raise ValueError("video must have shape [B,3,T,H,W]")
    if skin_mask is None:
        return video.mean(dim=(3, 4))
    if skin_mask.ndim != 5 or skin_mask.shape[1] != 1:
        raise ValueError("skin_mask must have shape [B,1,T,H,W]")
    mask = skin_mask.to(video).clamp(0.0, 1.0)
    numerator = (video * mask).sum(dim=(3, 4))
    denominator = mask.sum(dim=(3, 4)).clamp_min(eps)
    return numerator / denominator


def bandpass_fft(
    signal: Tensor,
    fs: float,
    low_hz: float,
    high_hz: float,
) -> Tensor:
    """沿最后一维做零相位理想带通，作为 ``scipy.filtfilt`` 的可微替代。

    通过 rFFT 施加频域硬掩码再逆变换。相比 Butterworth 是更陡峭的理想
    带通，但对"约束编辑信号落在生理频带"这一监督目的足够，且完全可微、
    零相位（不引入时延），适合与目标波形逐点比较。
    """
    length = signal.shape[-1]
    spectrum = torch.fft.rfft(signal, dim=-1)
    freqs = torch.fft.rfftfreq(length, d=1.0 / float(fs)).to(signal.device)
    band = ((freqs >= low_hz) & (freqs <= high_hz)).to(spectrum.real.dtype)
    filtered = spectrum * band
    return torch.fft.irfft(filtered, n=length, dim=-1)


# detrend 线性算子只依赖 (length, lambda, device, dtype)，缓存避免重复求逆。
_DETREND_CACHE: dict = {}


def _detrend_operator(
    length: int,
    lambda_value: float,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    """构造 Tarvainen 平滑先验去趋势的固定线性算子 ``M``。

    去趋势结果为 ``M @ x``，其中
    ``M = I - (I + lambda^2 D^T D)^{-1}``，``D`` 是二阶差分矩阵。
    该算子只与信号长度和 lambda 有关，因此按 key 缓存复用。
    """
    key = (int(length), float(lambda_value), device, dtype)
    operator = _DETREND_CACHE.get(key)
    if operator is not None:
        return operator
    identity = torch.eye(length, device=device, dtype=dtype)
    second_diff = torch.zeros(length - 2, length, device=device, dtype=dtype)
    rows = torch.arange(length - 2, device=device)
    second_diff[rows, rows] = 1.0
    second_diff[rows, rows + 1] = -2.0
    second_diff[rows, rows + 2] = 1.0
    regularizer = identity + (lambda_value ** 2) * (
        second_diff.transpose(0, 1) @ second_diff
    )
    operator = identity - torch.linalg.inv(regularizer)
    _DETREND_CACHE[key] = operator
    return operator


def detrend(signal: Tensor, lambda_value: float = 100.0) -> Tensor:
    """对 ``[B,T]`` 信号做可微去趋势，与 numpy ``utils.detrend`` 数学一致。"""
    if signal.ndim != 2:
        raise ValueError("detrend expects a [B,T] signal")
    operator = _detrend_operator(
        signal.shape[-1], lambda_value, signal.device, signal.dtype
    )
    return signal @ operator.transpose(0, 1)


def sliding_windows(
    rgb: Tensor, window_length: int, step: int
) -> Tuple[Tensor, int]:
    """把 ``[B,3,N]`` 的 RGB 序列切成滑动窗口 ``[B,3,num_windows,window_length]``。"""
    if rgb.shape[-1] < window_length:
        raise ValueError(
            "signal length must be >= window_length for windowing"
        )
    windows = rgb.unfold(2, window_length, step)
    return windows, windows.shape[2]
