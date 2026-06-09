import math

import torch
import torch.nn.functional as F
import torchaudio.functional as AF
from scipy import signal


def rppg_pos_torch(
    vid_bcthw: torch.Tensor,
    fs: float = 30.0,
    win_sec: float = 1.6,
    detrend_lambda: float = 100.0,
    normalize_output: bool = False,
    exact_legacy_windowing: bool = False,
) -> torch.Tensor:
    """
    尽可能逼近 POS_WANG 的可微 PyTorch 实现。

    输入:
        vid_btchw: [B,C,T,H,W]，建议 RGB 顺序
    输出:
        [B,T]

    exact_legacy_windowing:
        False: 包含最后一个完整窗口，推荐使用。
        True: 复现 POS_WANG 的窗口范围，最后一个完整窗口不会被处理。
    """
    if vid_bcthw.ndim != 5:
        raise ValueError("Expected input shape [B,C,T,H,W]")

    batch_size, channels, time_steps, _, _ = vid_bcthw.shape

    if channels < 3:
        raise ValueError("POS requires at least three RGB channels")

    if time_steps < 3:
        raise ValueError("POS requires at least three frames")

    x = vid_bcthw.clamp(0.0, 1.0)

    # 对应 _process_video：计算每帧 RGB 空间均值
    rgb = x[:, :3].mean(dim=(3, 4))  # [B,3,T]

    win_len = min(time_steps, max(2, math.ceil(win_sec * fs)))

    # [B,3,num_windows,win_len]
    windows = rgb.unfold(dimension=2, size=win_len, step=1)

    # 原始 POS_WANG 少处理最后一个完整窗口
    if exact_legacy_windowing and windows.shape[2] > 1:
        windows = windows[:, :, :-1, :]

    # 对应 RGB[m:n, :] / mean(RGB[m:n, :], axis=0)
    windows = windows / (
        windows.mean(dim=-1, keepdim=True).clamp_min(1e-6)
    )

    projection = x.new_tensor([
        [0.0, 1.0, -1.0],
        [-2.0, 1.0, 1.0],
    ])

    # [B,num_windows,2,win_len]
    projected = torch.einsum(
        "pc,bcnl->bnpl",
        projection,
        windows,
    )

    s0 = projected[:, :, 0, :]
    s1 = projected[:, :, 1, :]

    # np.std 默认使用总体标准差，因此 correction=0
    alpha = (
        s0.std(dim=-1, keepdim=True, correction=0)
        / s1.std(dim=-1, keepdim=True, correction=0).clamp_min(1e-6)
    )

    h = s0 + alpha * s1
    h = h - h.mean(dim=-1, keepdim=True)

    # 使用 fold 实现可微 overlap-add；不除以 overlap_count
    patches = h.transpose(1, 2)  # [B,win_len,num_windows]

    rppg = F.fold(
        patches,
        output_size=(1, time_steps),
        kernel_size=(1, win_len),
        stride=(1, 1),
    ).reshape(batch_size, time_steps)

    # 可微平滑先验 detrend，对应 utils.detrend(..., 100)
    rppg = _detrend_torch(rppg, detrend_lambda)

    # Butterworth 系数不依赖输入，因此由 SciPy 计算不影响梯度传播
    b_np, a_np = signal.butter(
        1,
        [0.75 / fs * 2.0, 3.0 / fs * 2.0],
        btype="bandpass",
    )

    b = torch.as_tensor(b_np, device=x.device, dtype=x.dtype)
    a = torch.as_tensor(a_np, device=x.device, dtype=x.dtype)

    # 可微零相位滤波，对应 scipy.signal.filtfilt
    rppg = AF.filtfilt(rppg, a_coeffs=a, b_coeffs=b, clamp=False)

    if normalize_output:
        rppg = rppg - rppg.mean(dim=-1, keepdim=True)
        rppg = rppg / (
            rppg.std(dim=-1, keepdim=True, correction=0) + 1e-6
        )

    return rppg


def _detrend_torch(
    signal_bt: torch.Tensor,
    regularization: float = 100.0,
) -> torch.Tensor:
    """
    可微平滑先验 detrend。

    对应常见实现：
        (I - inv(I + lambda^2 * D2.T @ D2)) @ signal
    """
    _, time_steps = signal_bt.shape
    device = signal_bt.device
    dtype = signal_bt.dtype

    identity = torch.eye(time_steps, device=device, dtype=dtype)

    second_difference = torch.zeros(
        time_steps - 2,
        time_steps,
        device=device,
        dtype=dtype,
    )

    row = torch.arange(time_steps - 2, device=device)
    second_difference[row, row] = 1.0
    second_difference[row, row + 1] = -2.0
    second_difference[row, row + 2] = 1.0

    system = identity + (
        regularization ** 2
    ) * (second_difference.transpose(0, 1) @ second_difference)

    trend = torch.linalg.solve(
        system,
        signal_bt.transpose(0, 1),
    ).transpose(0, 1)

    return signal_bt - trend