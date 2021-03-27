# -*- coding: utf-8 -*-
from typing import Tuple

import numpy as np
from scipy.signal import lfilter


def enframe(signal, nw, inc) -> np.ndarray:
    """将音频信号转化为帧。
    参数含义：
    signal:原始音频型号
    nw:每一帧的长度(这里指采样点的长度，即采样频率乘以时间间隔)
    inc:相邻帧的间隔（同上定义）
    """
    # 信号总长度
    signal_length = len(signal)
    # 若信号长度小于一个帧的长度，则帧数定义为1
    if signal_length <= nw:
        nf = 1
    else:  # 否则，计算帧的总长度
        nf = int(np.ceil((1.0 * signal_length - nw + inc) / inc))
    # 所有帧加起来总的铺平后的长度
    pad_length = int((nf - 1) * inc + nw)
    # 不够的长度使用0填补，类似于FFT中的扩充数组操作
    zeros = np.zeros((pad_length - signal_length,))
    # 填补后的信号记为pad_signal
    pad_signal = np.concatenate((signal, zeros))
    # 相当于对所有帧的时间点进行抽取，得到nf*nw长度的矩阵
    indices = np.tile(np.arange(0, nw), (nf, 1)) + np.tile(np.arange(0, nf * inc, inc), (nw, 1)).T
    # 将indices转化为矩阵
    indices = np.array(indices, dtype=np.int32)
    # 得到帧信号
    frames = pad_signal[indices]
    return frames


def vad(x: np.ndarray) -> Tuple:
    # 幅度归一化到[-1,1]
    x = x / np.abs(x).max()

    # 常数设置
    frame_len = 240
    frame_inc = 80

    amp1 = 10
    amp2 = 2
    zcr2 = 5

    # 6*10ms  = 30ms default 8
    max_silence = 300
    # 15*10ms = 150ms default 15
    min_len = 15
    status = 0
    count = 0
    silence = 0

    # 计算过零率
    tmp1 = enframe(x[:-2], frame_len, frame_inc)
    tmp2 = enframe(x[1:], frame_len, frame_inc)
    signs = (tmp1 * tmp2) < 0
    diffs = (tmp1 - tmp2) > 0.02
    zcr = np.sum(signs * diffs, 1)

    # 计算短时能量
    amp = np.sum(np.abs(enframe(np.array(lfilter([1, -0.9375], 1, x)), frame_len, frame_inc)), 1)

    # 调整能量门限
    amp1 = np.minimum(amp1, amp.max(0) / 4)
    amp2 = np.minimum(amp2, amp.max(0) / 8)

    # 开始端点检测
    x1 = 0
    x2 = 0
    for n in range(zcr.size):
        # 0 = 静音, 1 = 可能开始
        if status in [0, 1]:
            if amp[n] > amp1:  # 确信进入语音段
                x1 = max(n - count, 1)
                status = 2
                silence = 0
                count = count + 1
            elif amp[n] > amp2 or zcr[n] > zcr2:
                status = 1
                count = count + 1
            else:  # 静音状态
                status = 0
                count = 0
        elif status == 2:  # 2 = 语音段
            if amp[n] > amp2 or zcr[n] > zcr2:
                count = count + 1
            else:  # 语音将结束
                silence = silence + 1
                if silence < max_silence:  # 静音还不够长，尚未结束
                    count = count + 1
                elif count < min_len:  # 语音长度太短，认为是噪声
                    status = 0
                    silence = 0
                    count = 0
                else:  # 语音结束
                    status = 3
        else:
            break

    count = int(count - silence / 2)
    x2 = x1 + count - 1
    left = x1 * frame_inc - 1
    right = x2 * frame_inc

    return int(left), int(right)
