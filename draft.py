import numpy as np


a = np.array([[1, 2], [3, 4]])
b = np.array([[5, 0], [5, 1]])
c = np.array([[2, 6], [8, 0]])

x = np.array([a, b, c])

print(np.argmax(x, axis=0) + 1)

print(x)