import numpy as np
import timeit

def euclidean_distance(pointA: np.array, pointB: np.array):
    return np.sum(np.square(pointA - pointB))

def euclidean_squared_distance(pointA, pointB):
    """
    Compute the Euclidean squared distance between two points.
    
    Args:
        pointA (np.ndarray): First point.
        pointB (np.ndarray): Second point.
    
    Returns:
        float: The squared Euclidean distance.
    """
    return np.sum((pointA - pointB)**2)

def measure_construction_performance(ds, dimension: int, points: list, 
                                    dist_function: callable = None):
    """
    Measures the time taken to construct a GeometricDataStructure instance.
    
    Args:
        ds: The geometric data structure instance (e.g., BallTree, KDTree).
        dimension (int): Dimension of each point.
        points (list): List of points (each as a np.ndarray or convertible to one).
        dist_function (callable, optional): Distance function to use; if None, default is used.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the construction time in seconds.
    """
    start = timeit.default_timer()
    ds_instance = ds(dimension, points, dist_function)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_insert_performance(ds, point: np.ndarray):
    """
    Measures the time taken to perform an insert operation on a GeometricDataStructure instance.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements insert(point).
        point (np.ndarray): The point to be inserted.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the time in seconds.
    """
    start = timeit.default_timer()
    ds.insert(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_get_knn_performance(ds, point: np.ndarray, k: int):
    """
    Measures the time taken for the get_knn method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_knn.
        point (np.ndarray): The target point for the query.
        k (int): Number of nearest neighbours to query.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the time in seconds.
    """
    start = timeit.default_timer()
    ds.get_knn(point, k)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_get_nearest_performance(ds, point: np.ndarray):
    """
    Measures the time taken for the get_nearest method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_nearest.
        point (np.ndarray): The target point for the query.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the time in seconds.
    """
    start = timeit.default_timer()
    ds.get_nearest(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_delete_performance(ds, point: np.ndarray):
    """
    Measures the time taken for the delete method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements delete.
        point (np.ndarray): The point to delete.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the time in seconds.
    """
    start = timeit.default_timer()
    ds.delete(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_range_search_performance(ds, point: np.ndarray, radius: float):
    """
    Measures the time taken for the range search method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements range search.
        point (np.ndarray): The target point for the query.
        radius (float): The radius within which to search for points.

    Returns:
        tuple: (elapsed_time) where elapsed_time is the time in seconds.
    """
    start = timeit.default_timer()
    ds.query_range(point, radius)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_space_usage(ds):
    """
    Measures the memory usage of the data structure (in bytes).
    """
    return ds.__sizeof__()
