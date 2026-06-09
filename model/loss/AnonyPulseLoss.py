import torch
import re

def frequency_loss(signal, target_hr_prompt, fs=30):
    """
    signal: [B, T] rPPG 信号
    target_hr_prompt: List[str] or str，如 "heart rate 75 bpms"
    fs: 采样率 (Hz)
    """

    # === 提取数值 target_hr ===
    if isinstance(target_hr_prompt, str):
        match = re.search(r'\d+', target_hr_prompt)
        target_hr = torch.tensor([float(match.group()) if match else 0.0], device=signal.device)
    elif isinstance(target_hr_prompt, list):
        hr_list = []
        for s in target_hr_prompt:
            match = re.search(r'\d+', s)
            hr_list.append(float(match.group()) if match else 0.0)
        target_hr = torch.tensor(hr_list, device=signal.device)
    else:
        raise ValueError("target_hr_prompt 应该是 str 或 list[str]")
    
    # === 频谱主峰分析 ===
    B, T = signal.shape
    freqs = torch.fft.rfft(signal, dim=1)     # [B, T//2+1]
    magnitude = torch.abs(freqs)              # [B, T//2+1]
    peak_idx = torch.argmax(magnitude, dim=1) # [B]
    peak_freq_hz = peak_idx * fs / T          # [B]
    peak_hr = peak_freq_hz * 60               # [BPM]

    return torch.mean((peak_hr - target_hr) ** 2)
