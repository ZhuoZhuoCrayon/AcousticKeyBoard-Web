import math

import numpy as np
from scipy.signal import lfilter

from apps.keyboard.core.mfcc import trifbank, vec2frames


def nextpow2(n):
    return np.ceil(np.log2(np.abs(n))).astype("long")


# function [CC, FBE, frames] = mfcc(speech, fs, Tw, Ts, alpha, window, R, M, N, L)


def mfcc(speech, fs, Tw, Ts, alpha, window, R, M, N, L):

    # % Explode samples to the range of 16 bit shorts
    if np.abs(speech).max(0) <= 1:
        speech = speech * (pow(2, 15))

    Nw = round(1e-3 * Tw * fs)
    Ns = round(1e-3 * Ts * fs)

    nfft = pow(2, nextpow2(Nw))
    K = nfft / 2 + 1

    def hz2mel(hz):
        """hz2mel = @(hz)(1127 * log(1 + hz / 700));"""
        return 1127 * np.log(1 + hz / 700)

    def mel2hz(mel):
        """mel2hz = @(mel)(700 * exp(mel / 1127) - 700);"""
        return 700 * np.exp(mel / 1127) - 700

    def dctm(N, M):
        """sqrt(2.0 / M) * cos(
            repmat([0:N - 1].', 1, M) .* repmat(pi * ([1:M] - 0.5) / M, N, 1)
        )
        """
        return math.sqrt(2.0 / M) * np.cos(
            np.kron(np.ones((1, M)), np.arange(N).reshape(-1, 1))
            * np.kron(np.ones((N, 1)), np.pi * (np.arange(1, M + 1) - 0.5) / M)
        )

    def ceplifter(N, L):
        """1 + 0.5 * L * sin(pi * [0:N - 1] / L)"""
        return 1 + 0.5 * L * np.sin(np.pi * np.arange(N) / L)

    # speech = filter([1, -alpha], 1, speech);
    speech = lfilter([1, -alpha], 1, speech)

    frames = vec2frames.vec2frames(speech, Nw, Ns, "cols", window, False)

    # MAG = abs(fft(frames, nfft, 1));
    MAG = np.abs(np.fft.fft(frames, nfft, 0))

    H = trifbank.trifbank(M, K, R, fs, hz2mel, mel2hz)

    FBE = H @ MAG[0 : int(K)]

    DCT = dctm(N, M)

    CC = DCT @ np.log(FBE)

    lifter = ceplifter(N, L)

    CC = np.diag(lifter) @ CC

    return CC, FBE, frames
