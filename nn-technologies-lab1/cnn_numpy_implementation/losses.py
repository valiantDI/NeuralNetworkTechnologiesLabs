import numpy as np


def cross_entropy(y_true, y_pred):
    return -np.sum(y_true * np.log(y_pred))


def cross_entropy_prime(y_true, y_pred):
    return y_pred - y_true
