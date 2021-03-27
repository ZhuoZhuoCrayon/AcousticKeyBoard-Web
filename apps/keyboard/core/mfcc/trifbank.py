import numpy as np

# function [H, f, c] = trifbank(M, K, R, fs, h2w, w2h)


def trifbank(M, K, R, fs, h2w, w2h):

    f_min = 0
    f_low = R[0]
    f_high = R[1]
    f_max = 0.5 * fs
    f = np.linspace(f_min, f_max, int(K))
    h2w(f)

    # c = w2h(h2w(f_low) + [0:M + 1] * ((h2w(f_high) - h2w(f_low)) / (M + 1)) )
    c = w2h(h2w(f_low) + np.arange(M + 2) * ((h2w(f_high) - h2w(f_low)) / (M + 1)))
    h2w(c)

    H = np.zeros((int(M), int(K)))
    for m in range(1, M + 1):
        k = np.logical_and(f >= c[m - 1], f <= c[m])
        H[m - 1, k] = (f[k] - c[m - 1]) / (c[m] - c[m - 1])
        k = np.logical_and(f >= c[m], f <= c[m + 1])
        H[m - 1, k] = (c[m + 1] - f[k]) / (c[m + 1] - c[m])

    return H
