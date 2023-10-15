import numpy as np
from distance_functions import haversine_distance


def euclidean_grad(x, y):
    return x - y


def haversine_grad(x, y, R=6371):
    grad = np.zeros_like(x)
    a = haversine_distance(x, y)[1]
    grad[0] = R / (2 * np.sqrt(1 - a) * np.sqrt(a)) * np.sin(x[0] - y[0])
    grad[1] = (
        R
        / (2 * np.sqrt(1 - a) * np.sqrt(a))
        * np.cos(x[0])
        * np.cos(y[0])
        * np.sin(x[1] - y[1])
    )
    return grad
