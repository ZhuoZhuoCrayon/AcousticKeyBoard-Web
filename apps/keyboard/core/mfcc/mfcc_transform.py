import numpy as np

from apps.keyboard.core.mfcc import mfcc


def mapminmax(x, ymin=-1, ymax=+1):
    x = np.asanyarray(x)
    xmax = x.max(axis=-1)
    xmin = x.min(axis=-1)
    if (xmax == xmin).any():
        raise ValueError("some rows have no variation")
    return (ymax - ymin) * (x - xmin) / (xmax - xmin) + ymin


# function [Mfcc] = mfcc_transform(OriginalData, fs, Tw, Ts, alpha, M, C, HF, LF)
# % Define variables
# % Tw = 25;                % analysis frame duration (ms)   帧持续时长
# % Ts = 10;                % analysis frame shift (ms)    帧移
# % alpha = 0.97;           % preemphasis coefficient   预加重
# % M = 20;                 % number of filterbank channels  滤波器通道数
# % C = 13;                 % number of cepstral coefficients  倒谱系数
# L = 22; % cepstral sine lifter parameter   倒谱正弦参数
# % LF = 10;               % lower frequency limit (Hz)   低频门限
# % HF = 800;              % upper frequency limit (Hz)  高频门限


def mfcc_transform(OriginalData, fs, Tw, Ts, alpha, M, C, HF, LF):
    speech = mapminmax(OriginalData)
    # cepstral sine lifter parameter   倒谱正弦参数
    L = 22
    MFCCs, FBEs, frames = mfcc.mfcc(speech, fs, Tw, Ts, alpha, np.hamming, [LF, HF], M, C, L)
    return MFCCs.T
