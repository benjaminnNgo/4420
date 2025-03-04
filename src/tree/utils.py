
import numpy as np

def euclidean_squ_distance(pointA: np.array, pointB: np.array):
    return np.sum(np.square(pointA - pointB))
