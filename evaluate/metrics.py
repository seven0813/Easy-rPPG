import numpy as np
import torch

from .dai_post_process import calculate_HR


def cal_metric_liang(pred_phys: np.ndarray, label_phys: np.ndarray,methods=None) -> list:
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
            ret.append((np.abs((diff) / label_phys)).mean() * 100)
        elif m == "R":
            temp = np.corrcoef(pred_phys, label_phys)
            if np.isnan(temp).any() or np.isinf(temp).any():
                ret.append(-1 * np.ones(1))
            else:
                ret.append(temp[0, 1])
    return ret



def get_metrics(HR_pred, HR_real):
    
    HR_pred = np.array(HR_pred).reshape(-1)
    HR_real = np.array(HR_real).reshape(-1)
    temp = HR_pred - HR_real
    ME = np.mean(temp)
    STD = np.std(temp)
    MAE = np.sum(np.abs(temp)) / len(temp)
    RMSE = np.sqrt(np.sum(np.power(temp, 2)) / len(temp))
    MER = np.mean(np.abs(temp) / HR_real)
    
    ## 这个person计算公式不标注，分母部分加了个0.01，为了防止分母为0
    ## p = np.sum((HR_pred - np.mean(HR_pred)) * (HR_real - np.mean(HR_real))) / (0.01 + np.linalg.norm(HR_pred - np.mean(HR_pred), ord=2) * np.linalg.norm(HR_real - np.mean(HR_real), ord=2))
    pearson = np.corrcoef(HR_pred, HR_real)
    if np.isnan(pearson).any() or np.isinf(pearson).any():
        P = -1 
    else:
        P = pearson[0, 1]
    
    return ME, STD, MAE, RMSE, MER, P



def calculate_metrics(predictions, labels, config):
    '''
        三个评测级别：
            1. video：把一个视频的所有预测片段拼接起来，计算一次心率，得到一个预测心率和一个真实心率，计算一次指标。
            2. clip：每个片段单独计算心率，得到多个预测心率和多个真实心率，计算多个指标。
            3. clip_average：每个片段单独计算心率，得到多个预测心率和多个真实心率，对每个视频的预测心率和真实心率分别取平均，得到一个预测心率和一个真实心率，计算一次指标。
    '''
    hr_pred_list = []
    hr_bvp_list = []

    eval_level = config.inference.eval_level
    fs = config.datasets.fs
    
    for subj_index in predictions:
        sort_indices = sorted(predictions[subj_index])

        if eval_level == "video":
            pred_bvp = torch.cat(
                [predictions[subj_index][i] for i in sort_indices]
            )
            test_bvp = torch.cat(
                [labels[subj_index][i] for i in sort_indices]
            )

            hr_pred = float(calculate_HR(pred_bvp.numpy(), fs)[0])
            hr_bvp = float(calculate_HR(test_bvp.numpy(), fs)[0])
            hr_pred_list.append(hr_pred)
            hr_bvp_list.append(hr_bvp)

        elif eval_level in ("clip", "clip_average"):
            clip_hr_pred_list = []
            clip_hr_bvp_list = []

            for sort_index in sort_indices:
                pred_bvp = predictions[subj_index][sort_index]
                test_bvp = labels[subj_index][sort_index]
                hr_pred = float(calculate_HR(pred_bvp.numpy(), fs)[0])
                hr_bvp = float(calculate_HR(test_bvp.numpy(), fs)[0])

                clip_hr_pred_list.append(hr_pred)
                clip_hr_bvp_list.append(hr_bvp)

            if eval_level == "clip":
                hr_pred_list.extend(clip_hr_pred_list)
                hr_bvp_list.extend(clip_hr_bvp_list)
            else:
                hr_pred_list.append(float(np.mean(clip_hr_pred_list)))
                hr_bvp_list.append(float(np.mean(clip_hr_bvp_list)))

        else:
            raise ValueError("inference.eval_level must be video, clip, or clip_average.")        
    
    ME, STD, MAE, RMSE, MER, P = get_metrics(hr_pred_list, hr_bvp_list)
    
    return ME, STD, MAE, RMSE, MER, P      
