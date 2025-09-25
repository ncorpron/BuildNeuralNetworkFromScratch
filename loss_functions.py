import numpy as np

def mean_squared_error(y_true, y_pred):
    return np.mean((np.array(y_true) - np.array(y_pred)) ** 2)

def cross_entropy(y_true, y_pred, epsilon=1e-12):
    y_pred = np.clip(y_pred, epsilon, 1. - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
