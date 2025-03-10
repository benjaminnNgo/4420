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

def measure_construction_performance(DSClass, dimension: int, points: list, 
                                       dist_function: callable = None, trials: int = 1):
    """
    Measures the performance of constructing a GeometricDataStructure.
    
    Args:
        DSClass: The class of the geometric data structure (e.g., BallTree, KDTree).
        dimension (int): Dimension of each point.
        points (list): List of points (each a np.ndarray or convertible to one).
        dist_function (callable, optional): Distance function to use; if None, default is used.
        trials (int): Number of times to construct the data structure for averaging.
    
    Returns:
        tuple: (elapsed_time, space_used, ds) where elapsed_time is the average construction
               time in seconds, space_used is a shallow measurement in bytes of the constructed
               data structure, and ds is the data structure instance from the final trial.
    """
    import timeit, sys
    total_time = 0
    ds = None
    for _ in range(trials):
        start = timeit.default_timer()
        ds = DSClass(dimension, points, dist_function)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    space_used = sys.getsizeof(ds)
    return elapsed_time, space_used, ds

def measure_insert_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the time taken to perform an insert operation on a GeometricDataStructure instance
    and returns the average time taken, along with a shallow measurement of the space used.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements insert(point).
        point (np.ndarray): The point to be inserted.
        trials (int): Number of times to run the insert for averaging.
    
    Returns:
        tuple: (elapsed_time, space_used) where elapsed_time is the average time (s)
               and space_used (bytes) is measured using sys.getsizeof.
    """
    import timeit, sys
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.insert(point)
        total_time += timeit.default_timer() - start

    elapsed_time = total_time / trials
    space_used = sys.getsizeof(ds)
    return elapsed_time, space_used

def measure_get_knn_performance(ds, point: np.ndarray, k: int, trials: int = 1):
    """
    Measures the performance of the get_knn method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_knn.
        point (np.ndarray): The target point for the query.
        k (int): Number of nearest neighbours to query.
        trials (int): Number of times to run the query, to compute an average.
    
    Returns:
        tuple: (elapsed_time, space_used) where elapsed_time is the average time (s)
               and space_used (bytes) is measured using sys.getsizeof.
    """
    import timeit, sys
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.get_knn(point, k)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    space_used = sys.getsizeof(ds)
    return elapsed_time, space_used

def measure_get_nearest_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the performance of the get_nearest method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements get_nearest.
        point (np.ndarray): The target point for the query.
        trials (int): Number of times to run the query, to compute an average.
    
    Returns:
        tuple: (elapsed_time, space_used) where elapsed_time is the average time (s)
               and space_used (bytes) is measured using sys.getsizeof.
    """
    import timeit, sys
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.get_nearest(point)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    space_used = sys.getsizeof(ds)
    return elapsed_time, space_used

def measure_delete_performance(ds, point: np.ndarray, trials: int = 1):
    """
    Measures the performance of the delete method.
    
    Args:
        ds: Instance of a GeometricDataStructure that implements delete.
        point (np.ndarray): The point to delete.
        trials (int): Number of times to run the deletion, to compute an average.
    
    Returns:
        tuple: (elapsed_time, space_used) where elapsed_time is the average time (s)
               and space_used (bytes) is measured using sys.getsizeof.
    Note:
        Since deletion modifies the data structure, you may need to reinsert the point
        between trials if you intend to measure deletion repeatedly.
    """
    import timeit, sys
    total_time = 0
    for _ in range(trials):
        start = timeit.default_timer()
        ds.delete(point)
        total_time += timeit.default_timer() - start
    elapsed_time = total_time / trials
    space_used = sys.getsizeof(ds)
    return elapsed_time, space_used
