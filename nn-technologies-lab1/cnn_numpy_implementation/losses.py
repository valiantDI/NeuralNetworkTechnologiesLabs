import numpy as np


def binary_cross_entropy(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


def binary_cross_entropy_prime(y_true, y_pred):
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / np.size(y_true)


def cross_entropy(y_true, y_pred):
    # eps = np.finfo(float).eps
    return -np.sum(y_true * np.log(y_pred)) # + eps))


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
