import numpy as np


def grid(n):
    """Return the embedding of a grdi graph."""
    M = m ** 2
    x = np.linspace(0, 1, m)
    y = np.linspace(0, 1, m)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2))
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z
