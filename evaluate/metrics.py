import numpy as np
import torch

from .dai_post_process import calculate_HR as calculate_hr_dai
from .toolbox_post_process import calculate_HR as calculate_hr_toolbox


_MIN_POSTPROCESS_FRAMES = 9


def _canonical_label_type(value):
    if not isinstance(value, str):
        raise TypeError(
            "datasets.label_type must be a string, "
            f"got {type(value).__name__}."
        )
    aliases = {
        "raw": "Raw",
        "standardized": "Standardized",
        "diffnormalized": "DiffNormalized",
    }
    normalized = value.strip().replace("_", "").replace("-", "").lower()
    canonical = aliases.get(normalized)
    if canonical is None:
        raise ValueError(
            f"Unsupported datasets.label_type {value!r}. "
            "Supported: Raw, Standardized, DiffNormalized."
        )
    return canonical


def _resolve_hr_method(config):
    """Return None for Dai or FFT/Peak for the Toolbox postprocessor.

    Configuration:
        eval_method: Dai | Toolbox
        toolbox_hr_method: FFT | Peak
    """
    eval_method = str(
        config.inference.get("eval_method", "Dai")
    ).strip().lower()
    if eval_method == "dai":
        return None
    if eval_method != "toolbox":
        raise ValueError(
            "inference.eval_method must be Dai or Toolbox."
        )

    toolbox_method = str(
        config.inference.get("toolbox_hr_method", "FFT")
    ).strip().lower().replace("_", " ")
    if toolbox_method == "fft":
        return "FFT"
    if toolbox_method in {"peak", "peak detection"}:
        return "Peak"
    raise ValueError(
        "inference.toolbox_hr_method must be FFT or Peak/peak detection."
    )


def _to_numpy(waveform):
    if isinstance(waveform, torch.Tensor):
        waveform = waveform.detach().cpu().numpy()
    return np.asarray(waveform, dtype=np.float64).reshape(-1)


def _calculate_hr(waveform, fs, label_type, method):
    waveform = _to_numpy(waveform)
    if waveform.size < _MIN_POSTPROCESS_FRAMES:
        raise ValueError(
            f"Post-processing needs at least {_MIN_POSTPROCESS_FRAMES} frames, "
            f"got {waveform.size}."
        )
    diff_flag = label_type == "DiffNormalized"
    if method is None:
        # Dai 的 CWT 路径接收恢复后的完整波形。
        if diff_flag:
            waveform = np.cumsum(waveform)
        heart_rate = float(calculate_hr_dai(waveform, fs)[0])
    else:
        heart_rate = float(
            calculate_hr_toolbox(
                waveform,
                fs,
                diff_flag=diff_flag,
                hr_method=method,
            )[0]
        )
    if not np.isfinite(heart_rate):
        raise ValueError(
            f"{method or 'Dai'} post-processing produced a non-finite HR."
        )
    return heart_rate


def _iter_evaluation_segments(prediction, label, eval_level, eval_window=None):
    """Yield one full video or non-overlapping fixed-frame windows."""
    if len(prediction) != len(label):
        raise ValueError("Video prediction and label must have equal length.")
    if len(prediction) < _MIN_POSTPROCESS_FRAMES:
        raise ValueError(
            f"Video evaluation needs at least {_MIN_POSTPROCESS_FRAMES} frames."
        )
    if eval_level == "video":
        yield prediction, label
        return

    if eval_level != "window":
        raise ValueError("eval_level must be video or window.")
    if isinstance(eval_window, bool) or not isinstance(
        eval_window, (int, np.integer)
    ):
        raise ValueError(
            "inference.eval_window must be an integer number of frames."
        )
    if eval_window < _MIN_POSTPROCESS_FRAMES:
        raise ValueError(
            "inference.eval_window must be at least "
            f"{_MIN_POSTPROCESS_FRAMES} frames."
        )
    for start in range(0, len(prediction), eval_window):
        pred_window = prediction[start : start + eval_window]
        label_window = label[start : start + eval_window]
        if len(pred_window) >= _MIN_POSTPROCESS_FRAMES:
            yield pred_window, label_window


def _resolve_eval_levels(config):
    """Validate and return evaluation levels in configured execution order."""
    if "eval_levels" not in config.inference:
        raise ValueError("inference.eval_levels is required.")
    configured = config.inference.eval_levels
    if not isinstance(configured, (list, tuple)) or not configured:
        raise ValueError(
            "inference.eval_levels must be a non-empty list."
        )

    supported = {"video", "window", "window-average"}
    levels = []
    for value in configured:
        if not isinstance(value, str):
            raise ValueError(
                "Each inference.eval_levels item must be a string."
            )
        level = value.strip().lower()
        if level not in supported:
            raise ValueError(
                "inference.eval_levels items must be video, window, "
                "or window-average."
            )
        if level in levels:
            raise ValueError(
                f"inference.eval_levels contains duplicate level {level!r}."
            )
        levels.append(level)
    return levels


def _resolve_eval_protocol(config, configured):
    """Resolve one configured video/window/window-average protocol."""
    if configured == "video":
        return "video", None
    if configured in {"window", "window-average"}:
        if "eval_window" not in config.inference:
            raise ValueError(
                "inference.eval_window is required when inference.eval_levels "
                f"includes {configured}."
            )
        return configured.replace("-", "_"), config.inference.eval_window
    raise ValueError(
        "inference.eval_levels items must be video, window, or window-average."
    )



def get_metrics(HR_pred, HR_real):
    
    HR_pred = np.array(HR_pred).reshape(-1)
    HR_real = np.array(HR_real).reshape(-1)
    if HR_pred.size == 0 or HR_real.size == 0:
        raise ValueError("No valid HR samples were produced during evaluation.")
    if HR_pred.shape != HR_real.shape:
        raise ValueError("Predicted and ground-truth HR arrays must have equal shape.")

    temp = HR_pred - HR_real
    ME = np.mean(temp)
    STD = np.std(temp)
    MAE = np.sum(np.abs(temp)) / len(temp)
    RMSE = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    nonzero = np.abs(HR_real) > np.finfo(float).eps
    MER = (
        np.mean(np.abs(temp[nonzero]) / HR_real[nonzero])
        if np.any(nonzero)
        else np.nan
    )
    
    ## 这个person计算公式不标注，分母部分加了个0.01，为了防止分母为0
    ## p = np.sum((HR_pred - np.mean(HR_pred)) * (HR_real - np.mean(HR_real))) / (0.01 + np.linalg.norm(HR_pred - np.mean(HR_pred), ord=2) * np.linalg.norm(HR_real - np.mean(HR_real), ord=2))
    if len(HR_pred) < 2 or np.std(HR_pred) == 0 or np.std(HR_real) == 0:
        P = -1.0
    else:
        P = float(np.corrcoef(HR_pred, HR_real)[0, 1])
    
    return ME, STD, MAE, RMSE, MER, P



def _calculate_metrics_for_level(predictions, labels, config, configured_level):
    """Calculate HR metrics for one evaluation level.

    Model clips are sorted and concatenated per video before evaluation.
    ``eval_window`` is measured in frames. ``window-average`` averages the
    window HR values within each video before calculating dataset metrics.
    """
    hr_pred_list = []
    hr_bvp_list = []

    eval_level, eval_window = _resolve_eval_protocol(config, configured_level)
    fs = float(config.datasets.fs)
    label_type = _canonical_label_type(config.datasets.get("label_type", "Raw"))
    method = _resolve_hr_method(config)
    
    for subj_index in predictions:
        if subj_index not in labels:
            raise KeyError(f"Missing labels for subject {subj_index!r}.")
        sort_indices = sorted(predictions[subj_index])
        missing_label_clips = [
            i for i in sort_indices if i not in labels[subj_index]
        ]
        if missing_label_clips:
            raise KeyError(
                f"Missing label clips for subject {subj_index!r}: "
                f"{missing_label_clips[:5]}"
            )

        pred_bvp = torch.cat(
            [predictions[subj_index][i] for i in sort_indices]
        )
        test_bvp = torch.cat(
            [labels[subj_index][i] for i in sort_indices]
        )
        segment_level = "window" if eval_level == "window_average" else eval_level
        subject_hr_pred = []
        subject_hr_bvp = []
        for pred_window, label_window in _iter_evaluation_segments(
            _to_numpy(pred_bvp),
            _to_numpy(test_bvp),
            segment_level,
            eval_window,
        ):
            subject_hr_pred.append(
                _calculate_hr(pred_window, fs, label_type, method)
            )
            subject_hr_bvp.append(
                _calculate_hr(label_window, fs, label_type, method)
            )

        if eval_level == "window_average":
            hr_pred_list.append(float(np.mean(subject_hr_pred)))
            hr_bvp_list.append(float(np.mean(subject_hr_bvp)))
        else:
            hr_pred_list.extend(subject_hr_pred)
            hr_bvp_list.extend(subject_hr_bvp)
    
    ME, STD, MAE, RMSE, MER, P = get_metrics(hr_pred_list, hr_bvp_list)
    
    return ME, STD, MAE, RMSE, MER, P      


def calculate_metrics(predictions, labels, config):
    """Calculate every protocol listed in ``inference.eval_levels``.

    Model predictions are reused; only HR post-processing and metric
    aggregation are repeated for each configured level.
    """
    return {
        level: _calculate_metrics_for_level(
            predictions,
            labels,
            config,
            level,
        )
        for level in _resolve_eval_levels(config)
    }
