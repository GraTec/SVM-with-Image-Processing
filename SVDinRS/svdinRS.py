import numpy as np

# m = np.array([[1, 5, 2, 4, 2], [3, 4, 0, 3, 5], [
#              5, 0, 2, 3, 1], [1, 4, 0, 0, 1], [0, 2, 5, 3, 5]])

m = np.array([[1, 0, 0, 0, 2], [0, 0, 3, 0, 0], [
             0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 4, 0, 0, 0]])

u, sigma, vt = np.linalg.svd(m)

sigma = np.diag(sigma)

print(sigma)

print(u @ sigma @ vt)
