import numpy as np


def euclidean_distance(x, y):
    return np.linalg.norm(x - y)**2

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
    a = np.sin(dLat/2) * np.sin(dLat/2) + np.cos(lat1) \
        * np.cos(lat2) * np.sin(dLon/2) * np.sin(dLon/2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return R * c

def haversine_divergence(x, y, c=4):
    return np.linalg.norm(np.sin(x/c))**2 - np.linalg.norm(np.sin(y/c))**2 - (1/c*np.sin(2*y/c)).T.dot(x-y)

def great_circle_distance(x, y, R=6371):
    # Great circle distance between two points on a sphere
    # x and y are tuples with (latitude, longitude)
    # R is the radius of the Earth in kilometers
    
    lat1, lon1 = x
    lat2, lon2 = y

    d = np.arccos(np.sin(lat1)*np.sin(lat2)+np.cos(lat1)*np.cos(lat2)*np.cos(lon2-lon1))
    return R * d