import numpy as np

def euclidean_squ_distance(pointA: np.array, pointB: np.array):
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
                                       dist_function: callable = None, trials: int = 1):
    """
    Measures the time taken to construct a GeometricDataStructure instance.
    
    Args:
        ds: The geometric data structure instance (e.g., BallTree, KDTree).
        dimension (int): Dimension of each point.
        points (list): List of points (each as a np.ndarray or convertible to one).
        dist_function (callable, optional): Distance function to use; if None, default is used.
        trials (int): Number of times to run the construction for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average construction time in seconds.
    """
    import timeit
    total_time = 0
    ds = None
    for _ in range(trials):
        start = timeit.default_timer()
        ds = ds(dimension, points, dist_function)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    return elapsed_time

def measure_insert_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the time taken to perform an insert operation on a GeometricDataStructure instance.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements insert(point).
        point (np.ndarray): The point to be inserted.
        trials (int): Number of times to run the insert operation for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average time in seconds.
    """
    import timeit
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.insert(point)
        total_time += timeit.default_timer() - start

    elapsed_time = total_time / trials
    return elapsed_time

def measure_get_knn_performance(ds, point: np.ndarray, k: int, trials: int = 1):
    """
    Measures the time taken for the get_knn method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_knn.
        point (np.ndarray): The target point for the query.
        k (int): Number of nearest neighbours to query.
        trials (int): Number of times to run the query for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average time in seconds.
    """
    import timeit
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.get_knn(point, k)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    return elapsed_time

def measure_get_nearest_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the time taken for the get_nearest method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_nearest.
        point (np.ndarray): The target point for the query.
        trials (int): Number of times to run the query for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average time in seconds.
    """
    import timeit
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.get_nearest(point)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    return elapsed_time

def measure_delete_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the time taken for the delete method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements delete.
        point (np.ndarray): The point to delete.
        trials (int): Number of times to run the deletion for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average time in seconds.
    
    Note:
        Since deletion modifies the data structure, the point will be reinserted if
        another trial is needed.
    """
    import timeit
    total_time = 0
    for i in range(trials):
        start = timeit.default_timer()
        ds.delete(point)
        total_time += timeit.default_timer() - start

        # Reinsert the point if there are remaining trials
        if i < trials - 1:
            ds.insert(point)

    elapsed_time = total_time / trials
    return elapsed_time

def measure_range_search_performance(ds, point: np.ndarray, radius: float, trials: int = 1):
    """
    Measures the time taken for the range search method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements range search.
        point (np.ndarray): The target point for the query.
        radius (float): The radius within which to search for points.
        trials (int): Number of times to run the range search for averaging.
    
    Returns:
        tuple: (elapsed_time) where elapsed_time is the average time in seconds.
    """
    import timeit
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.range_search(point, radius)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    return elapsed_time

def measure_space_usage(ds):
    """
    Measures the memory usage of the data structure (in bytes).
    """
    return ds.__sizeof__()