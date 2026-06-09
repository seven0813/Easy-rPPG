import torch
import math
import numpy as np
import pycwt
from scipy import signal
from scipy.signal import butter, filtfilt


frequencies = [
               3.75, 3.720703125, 3.69140625, 3.662109375, 3.6328125, 3.603515625, 3.5742187500000004, 3.544921875,
               3.515625, 3.486328125, 3.45703125, 3.427734375, 3.3984375, 3.369140625, 3.33984375, 3.310546875,
               3.28125,
               3.251953125, 3.22265625, 3.193359375, 3.1640625, 3.134765625, 3.10546875, 3.076171875, 3.046875,
               3.017578125,
               2.98828125, 2.958984375, 2.9296875, 2.9003906249999996, 2.87109375, 2.841796875, 2.8125, 2.783203125,
               2.75390625, 2.7246093750000004, 2.6953125, 2.666015625, 2.6367187500000004, 2.607421875, 2.578125,
               2.548828125, 2.51953125, 2.490234375, 2.4609375, 2.431640625, 2.40234375, 2.373046875, 2.34375,
               2.314453125,
               2.28515625, 2.255859375, 2.2265625, 2.197265625, 2.16796875, 2.138671875, 2.109375, 2.080078125,
               2.05078125,
               2.021484375, 1.9921875, 1.962890625, 1.93359375, 1.904296875, 1.875, 1.845703125, 1.81640625,
               1.7871093750000002, 1.7578125, 1.728515625, 1.69921875, 1.669921875, 1.640625, 1.611328125,
               1.58203125,
               1.552734375, 1.5234375, 1.494140625, 1.46484375, 1.435546875, 1.40625, 1.376953125, 1.34765625,
               1.3183593750000002, 1.2890625, 1.259765625, 1.23046875, 1.201171875, 1.171875, 1.142578125,
               1.11328125,
               1.083984375, 1.0546875, 1.025390625, 0.99609375, 0.966796875, 0.9375, 0.908203125, 0.87890625,
               0.849609375,
               0.8203125, 0.791015625, 0.76171875, 0.732421875, 0.703125, 0.673828125, 0.64453125
               ]


def maxscale(array1):
    list1 = []
    for i in range(array1.shape[0]):
        list1.append(np.mean(array1[i]))
    return np.array(list1)


def rowcal(array1, row1):
    array2 = np.zeros(array1.shape)
    for i in range(array1.shape[1]):
        array2[:, i] = array1[:, i] * row1
    return array2


def cwt_filtering(listin, samplingrate, frequencies=frequencies):
    sr = samplingrate
    plf1 = np.array(listin)
    result = pycwt.cwt(plf1, 1 / sr, freqs=np.array(frequencies))
    cwtmatr = result[0]
    scale1 = maxscale(abs(result[0]))
    co = np.argmax(scale1)
    myguasswindow = np.array([0.0 for x in range(len(scale1))])
    for j in range(len(scale1)):
        myguasswindow[j] = math.exp(-1 * ((j - co) / (0.08 * len(scale1))) ** 2)
    mycwtmatr = rowcal(abs(result[0]), myguasswindow)
    mycwtmatr2 = rowcal(result[0].real, myguasswindow)
    result_copy = result[1][:]
    result3 = pycwt.icwt(mycwtmatr2, result_copy, 1 / sr).real
    return result3, mycwtmatr, cwtmatr



def peakcheckez(a, samplingrate):
    '''
        找时域波形里的局部峰值。
        计算相邻峰之间的采样点间隔 Δn。
        心率 = 60 * fs / Δn。
        把所有相邻峰间距算出的 bpm 再取平均。
    '''
    result = []
    for i in range(len(a)):
        if i == 0 or i == len(a) - 1:
            pass
        else:
            if a[i] >= a[i - 1] and a[i] > a[i + 1]:
                result.append(i)

    hr_list = []
    if len(result) <= 1:
        hr = 0
    else:
        for i in range(len(result) - 1):
            hr = 60 * samplingrate / (result[i + 1] - result[i])
            hr_list.append(hr)
        hr = np.mean(np.array(hr_list))
    return hr


# def calculate_HR(tmp, samplingrate=30):
#     f1 = 0.5
#     f2 = 3
#     samplingrate = samplingrate
#     ## 1. bandpass filter: 0.5Hz-3Hz (30-180 bpm)
#     b, a = signal.butter(4, [2 * f1 / samplingrate, 2 * f2 / samplingrate], "bandpass")
#     tmp = signal.filtfilt(b, a, np.array(tmp))
#     ## 2. 使用小波变换在预设频率列表中寻找能量最大的尺度，这一步的本质是“再去噪，再突出周期成分”
#     tmp = cwt_filtering(tmp, samplingrate)[0]

#     ## 3. 找时域波形里的局部峰值
#     hr_caled = peakcheckez(tmp, samplingrate)
#     return hr_caled, tmp



def butter_bandpass(sig, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter
    sig = np.reshape(sig, -1)
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype="band")

    y = filtfilt(b, a, sig)
    return y


def calculate_HR(sig, fs, lowcut=0.6, highcut=4, order=2):
    # Estimate HR from a signal using bandpass filtering, CWT, and peak detection.

    ## 0. 使用butterworth bandpass filter
    filter_tmp = butter_bandpass(sig, lowcut, highcut, fs, order)

    #? 此处为什么要使用双重滤波呢？
    ## 1.bandpass filter: 0.5Hz-3Hz (30-180 bpm)
    b, a = signal.butter(4, [2 * 0.5 / fs, 2 * 3 / fs], "bandpass")
    tmp = signal.filtfilt(b, a, np.array(filter_tmp))
    
    ## 2. 使用小波变换在预设频率列表中寻找能量最大的尺度，这一步的本质是“再去噪，再突出周期成分”
    tmp = cwt_filtering(tmp, fs)[0]

    ## 3. 找时域波形里的局部峰值
    hr_caled = peakcheckez(tmp, fs)

    return hr_caled, filter_tmp


def butter_bandpass_batch(sig_list, lowcut, highcut, fs, order=2):
    # butterworth bandpass filter (batch version)
    # signals are in the sig_list
    y_list = []

    for sig in sig_list:
        sig = np.reshape(sig, -1)
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        b, a = butter(order, [low, high], btype="band")
        y = filtfilt(b, a, sig)
        y_list.append(y)
    return np.array(y_list)

