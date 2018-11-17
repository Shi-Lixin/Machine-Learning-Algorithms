import numpy as np


def validate(X_data, y_data, ratio=0.15):
    N = X_data.shape[0]
    size = int(N * ratio)
    inds = np.random.permutation(range(N))
    for i in range(int(N / size)):
        test_ind = inds[i * size:(i + 1) * size]
        train_ind = list(set(range(N))-set(test_ind))
        yield X_data[train_ind], y_data[train_ind], X_data[test_ind], y_data[test_ind]
