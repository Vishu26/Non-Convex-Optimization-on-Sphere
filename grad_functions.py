import numpy as np
from distance_functions import haversine_distance


def euclidean_grad(x, y):
    return x - y


def haversine_grad(x, y, R=6371):
    grad = np.zeros_like(x)
    grad[0] = (
        2
        * R
        / ((1 - haversine_distance(x, y)[1]) * np.sqrt(haversine_distance(x, y)[1]))
        * np.sin(x[0] - y[0])
    )
    grad[1] = (
        2
        * R
        / ((1 - haversine_distance(x, y)[1]) * np.sqrt(haversine_distance(x, y)[1]))
        * np.cos(x[0])
        * np.sin(y[0])
        * np.sin(x[1] - y[1])
    )
    return grad
