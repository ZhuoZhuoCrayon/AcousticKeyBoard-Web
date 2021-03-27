import numpy as np

# function [frames, indexes] = vec2frames(vec, Nw, Ns, direction, window, padding)


def vec2frames(vec, Nw, Ns, direction="cols", window=np.hamming, padding=False):

    L = vec.shape[0]
    M = np.floor((L - Nw) / Ns + 1)

    E = L - ((M - 1) * Ns + Nw)

    if E > 0:
        P = Nw - E
        if isinstance(padding, bool) and padding:
            vec = np.vstack((vec, np.zeros(P, 1)))
        else:
            M = M - 1
        M = M + 1

    if direction == "rows":
        # indf = Ns * [0:(M - 1)].';
        indf = Ns * np.arange(M).reshape(-1, 1)
        # inds = [1:Nw];
        inds = np.arange(1, Nw + 1)
        # indexes = indf(:, ones(1, Nw)) + inds(ones(M, 1),:);
        indexes = np.kron(np.ones((1, Nw)), indf) + np.kron(np.ones((M, 1)), inds)

    elif direction == "cols":
        indf = Ns * np.arange(M)
        inds = np.arange(1, Nw + 1).reshape(-1, 1)
        indexes = np.kron(np.ones((int(Nw), 1)), indf) + np.kron(np.ones((1, int(M))), inds)
    else:
        print("Direction: %s not supported!\n", direction)

    frames = np.zeros(indexes.shape)
    for row_idx, index_list in enumerate(indexes):
        for col_idx, index in enumerate(index_list):
            frames[row_idx, col_idx] = vec[int(index) - 1]

    window = window(Nw)
    return frames @ np.diag(window) if direction == "rows" else np.diag(window) @ frames
