import numpy as np

A = np.array([
    [6, 24, 1],
    [13, 16, 10],
    [20, 17, 15]
], dtype=float)

inverse = np.linalg.inv(A)

print(np.round(inverse, 3))