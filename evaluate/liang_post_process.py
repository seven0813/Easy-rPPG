import numpy as np
import scipy
import scipy.io
from scipy.signal import butter, periodogram
from scipy.sparse import spdiags


def next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def detrend(signal, lambda_value=100):
    T = signal.shape[-1]
    H = np.identity(T)  # T x T
    ones = np.ones(T)  # T,
    minus_twos = -2 * np.ones(T)  # T,
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, (T - 2), T).toarray()
    designal = (H - np.linalg.inv(H + (lambda_value ** 2) * D.T.dot(D))).dot(signal.T).T
    return designal


def calculate_HR(signal: np.ndarray, fs=30, target="pulse", diff=False, detrend_flag=True):
    signal = np.asarray(signal)
    if signal.ndim == 2:
        signal = signal.reshape(-1)
        
    if diff:
        signal = signal.cumsum(axis=-1)
    if detrend_flag:
        signal = detrend(signal, 100)
    # get filter and detrend
    if target == "pulse":
        [b, a] = butter(1, [0.75 / fs * 2, 2.5 / fs * 2], btype='bandpass')
    else:
        [b, a] = butter(1, [0.08 / fs * 2, 0.5 / fs * 2], btype='bandpass')
    # bandpass
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # get psd
    N = next_power_of_2(signal.shape[-1])
    freq, psd = periodogram(signal, fs=fs, nfft=N, detrend=False)

    # get mask
    if target == "pulse":
        mask = np.argwhere((freq >= 0.75) & (freq <= 2.5))
    else:
        mask = np.argwhere((freq >= 0.08) & (freq <= 0.5))
    # get peak
    freq = freq[mask]
    if len(signal.shape) == 1:
        # phys = np.take(freq, np.argmax(np.take(psd, mask))) * 60
        idx = psd[mask.reshape(-1)].argmax(-1)
    else:
        idx = psd[:, mask.reshape(-1)].argmax(-1)
    phys = np.squeeze(freq[idx] * 60)
    return phys, signal


def cal_metric_liang(
    pred_phys: np.ndarray,
    label_phys: np.ndarray,
    methods=None,
) -> list:
    if methods is None:
        methods = ["Mean", "Std", "MAE", "RMSE", "MAPE", "R"]
    pred_phys = pred_phys.reshape(-1)
    label_phys = label_phys.reshape(-1)
    diff = pred_phys - label_phys
    ret = [] * len(methods)
    for m in methods:
        if m == "Mean":
            ret.append((diff).mean())
        elif m == "Std":
            ret.append((diff).std())
        elif m == "MAE":
            ret.append(np.abs(diff).mean())
        elif m == "RMSE":
            ret.append(np.sqrt((np.square(diff)).mean()))
        elif m == "MAPE":
            nonzero = np.abs(label_phys) > np.finfo(float).eps
            ret.append(
                np.mean(np.abs(diff[nonzero] / label_phys[nonzero])) * 100
                if np.any(nonzero)
                else np.nan
            )
        elif m == "R":
            if (
                len(pred_phys) < 2
                or np.std(pred_phys) == 0
                or np.std(label_phys) == 0
            ):
                ret.append(-1.0)
            else:
                ret.append(float(np.corrcoef(pred_phys, label_phys)[0, 1]))
    return ret
