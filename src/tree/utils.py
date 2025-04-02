"""
Utility functions for benchmarking geometric data structures.

This module provides functions for:
- Distance calculations (Euclidean and squared Euclidean)
- Performance measurement for common operations (construction, queries, modifications)
- Memory usage estimation

These utilities are designed to work with any class that implements the 
GeometricDataStructure interface.
"""

import numpy as np
import timeit

def euclidean_distance(pointA, pointB):
    """
    Compute the Euclidean distance between two points.
    
    Args:
        pointA (np.ndarray): First point.
        pointB (np.ndarray): Second point.
    
    Returns:
        float: The Euclidean distance.
    """
    return np.sqrt(np.sum(np.square(pointA - pointB)))

def euclidean_squared_distance(pointA, pointB):
    """
    Compute the Euclidean squared distance between two points.
    This is faster than euclidean_distance when only comparing distances
    (avoids the square root calculation).
    
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
    Uses timeit.default_timer for high-precision timing.
    
    Args:
        ds: The geometric data structure class (e.g., BallTree, KDTree).
        dimension (int): Dimension of each point.
        points (list): List of points (each as a np.ndarray or convertible to one).
        dist_function (callable, optional): Distance function to use; if None, default is used.

    Returns:
        float: Construction time in seconds.
    """
    start = timeit.default_timer()
    ds_instance = ds(dimension, points, dist_function)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_insert_performance(ds, point):
    """
    Measures the time taken to perform an insert operation on a GeometricDataStructure instance.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements insert(point).
        point (np.ndarray): The point to be inserted.

    Returns:
        float: Insert operation time in seconds.
    """
    start = timeit.default_timer()
    ds.insert(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_get_knn_performance(ds, point, k: int):
    """
    Measures the time taken for the get_knn method (k-nearest neighbors query).
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_knn.
        point (np.ndarray): The target point for the query.
        k (int): Number of nearest neighbours to retrieve.

    Returns:
        float: Query time in seconds.
    """
    start = timeit.default_timer()
    ds.get_knn(point, k)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_get_nearest_performance(ds, point):
    """
    Measures the time taken for the get_nearest method (single nearest neighbor query).
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_nearest.
        point (np.ndarray): The target point for the query.

    Returns:
        float: Query time in seconds.
    """
    start = timeit.default_timer()
    ds.get_nearest(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_delete_performance(ds, point):
    """
    Measures the time taken for the delete method.
    Note: For BallTree, this may use the lazy_delete implementation.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements delete.
        point (np.ndarray): The point to delete.

    Returns:
        float: Delete operation time in seconds.
    """
    start = timeit.default_timer()
    ds.delete(point)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_range_search_performance(ds, point, radius: float):
    """
    Measures the time taken for the range search method.
    Returns all points within the specified radius of the query point.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements query_range.
        point (np.ndarray): The target point for the query.
        radius (float): The radius within which to search for points.

    Returns:
        float: Query time in seconds.
    """
    start = timeit.default_timer()
    ds.query_range(point, radius)
    elapsed_time = timeit.default_timer() - start
    return elapsed_time

def measure_space_usage(ds):
    """
    Measures the memory usage of the data structure (in bytes).
    
    Uses the __sizeof__ method which should account for all components of the data structure,
    including numpy arrays, internal nodes, and other data structures.
    
    Args:
        ds: Instance of a GeometricDataStructure with __sizeof__ implementation.
    
    Returns:
        int: Memory usage in bytes.
    """
    return ds.__sizeof__()
