import numpy as np


def euclidean_distance(x, y):
    return 1 / 2 * np.linalg.norm(x - y) ** 2


def get_sines(x, i):
    if i < 0:
        return 1
    else:
        return np.multiply(np.sin(x[i]), get_sines(x, i - 1))


def euclidean_distance_polar(x, y, r):
    # Euclidean Distance in N dimensional polar coordinates
    dist = 0
    x = x + np.pi / 2
    x[-1] = x[-1] + np.pi / 2
    y = y + np.pi / 2
    y[-1] = y[-1] + np.pi / 2
    for i in range(len(x)):
        dist += (
            r**2
            * (np.cos(x[i]) * get_sines(x, i - 1) - np.cos(y[i]) * get_sines(y, i - 1))
            ** 2
        )
    dist += r**2 * (get_sines(x, len(x) - 1) - get_sines(y, len(y) - 1)) ** 2
    return 1 / 2 * dist


def manhattan_distance(x, y):
    return np.linalg.norm(x - y, ord=1)


def chebyshev_distance(x, y):
    return np.linalg.norm(x - y, ord=np.inf)


def minkowski_distance(x, y, p):
    return np.linalg.norm(x - y, ord=p)


def haversine_distance(x, y, R=6371):
    # Haversine distance between two points on a sphere
    # x and y are tuples with (latitude, longitude)
    # R is the radius of the Earth in kilometers

    lat1, lon1 = x
    lat2, lon2 = y
    dLat = lat2 - lat1
    dLon = lon2 - lon1
    a = np.sin(dLat / 2) * np.sin(dLat / 2) + np.cos(lat1) * np.cos(lat2) * np.sin(
        dLon / 2
    ) * np.sin(dLon / 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c, a


def haversine_divergence(x, y, c=5.83465):
    # Haversine Divergence in N dimensional polar coordinates
    return (
        np.linalg.norm(np.sin(x / c)) ** 2
        - np.linalg.norm(np.sin(y / c)) ** 2
        - (1 / c * np.sin(2 * y / c)).T.dot(x - y)
    )


def great_circle_distance(x, y, R=6371):
    # Great circle distance between two points on a sphere
    # x and y are tuples with (latitude, longitude)
    # R is the radius of the Earth in kilometers

    lat1, lon1 = x
    lat2, lon2 = y

    d = np.arccos(
        np.sin(lat1) * np.sin(lat2) + np.cos(lat1) * np.cos(lat2) * np.cos(lon2 - lon1)
    )
    return R * d
