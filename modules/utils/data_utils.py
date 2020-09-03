import numpy as np


def generate_proportion_data(max_rate=0.6, length=100, noise=0.2):
    """
    """
    X = np.array([i for i in range(length)])

    y = np.linspace(0.1, max_rate, length)
    y = y + np.random.normal(0, noise, length)
    y = np.clip(y, 0, 0.9)
    return X, y
