import numpy as np


def grid(m):
    """Return the embedding of a grdi graph."""
    M = m ** 2
    x = np.linspace(0, 1, m)
    print('x', x)
    y = np.linspace(0, 1, m)
    print('y', y)
    xx, yy = np.meshgrid(x, y)
    z = np.empty((M, 2))
    z[:, 0] = xx.reshape(M)
    z[:, 1] = yy.reshape(M)
    return z

print(grid(5))
