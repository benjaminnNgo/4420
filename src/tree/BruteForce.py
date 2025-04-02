import os
import sys
import numpy as np
from typing import Optional
from typing import *
import heapq

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure


"""
Authors: Tran Gia Bao Ngo
Overview: Brute force implementation of geometric data structure.

This implementation uses a simple set-based approach that stores all points 
and performs linear searches for queries. While not efficient for large datasets,
it provides a reliable baseline for comparison with more sophisticated data structures
like KD-Tree and Ball*-Tree.
"""
        
class BruteForce(GeometricDataStructure):
    r"""Brute force implementation of geometric data structure.
    
    This implementation maintains all points in a flat set structure and performs 
    linear scans for all operations. It serves as a baseline for performance comparison
    with more sophisticated spatial data structures.
    
    - Data is stored as a set of tuples for O(1) lookup and uniqueness checking
    - Query operations scan the entire dataset with time complexity O(n)
    - No spatial partitioning or preprocessing is performed

    Args:
        dimension (int): Dimension of each point
        points (List[np.ndarray]): List of initial points
        dist_function (callable): Function to compute distance between 2 points,
                                  by default is euclidean_squared_distance
    """
    def __init__(self, dimension: int,
                 points: Optional[List[np.ndarray]] = None,
                 dist_function: Optional[Callable] = None):
        """
        Initialize the brute force data structure.
        
        Converts the input numpy arrays to tuples for storage in a set,
        which allows for O(1) lookups and ensures point uniqueness.
        """
        super().__init__(dimension, points, dist_function)
        
        # Convert points to a set of tuples for efficient lookup and uniqueness
        self.points = self._to_set_of_tuple(self.points)

    def __sizeof__(self):
        """
        Calculate the memory usage of this data structure in bytes.
        
        Accounts for:
        - The dictionary of object attributes
        - The set container overhead
        - Each tuple's overhead
        - The actual float data in each point
        
        Returns:
            int: Memory usage in bytes
        """
        size = sys.getsizeof(self.__dict__)  # Get the size of the object attributes
        size += sys.getsizeof(self.points)   # Add the size of the points container (set)
        
        for point in self.points:
            size += sys.getsizeof(point)     # Tuple overhead
            size += 8 * len(point)           # Add 8 bytes per float value in the tuple
        
        return size
    
    def _to_set_of_tuple(self, list: List[np.ndarray]) -> Set[Tuple]:
        """
        Convert a list of numpy arrays to a set of tuples.
        
        This conversion is necessary because:
        1. numpy arrays are not hashable and cannot be stored in sets
        2. tuples are hashable and enable O(1) lookup
        3. sets ensure point uniqueness
        
        Args:
            list (List[np.ndarray]): List of points as numpy arrays
            
        Returns:
            Set[Tuple]: Set of points as tuples
        """
        if list is None:
            return set()
            
        set_of_points = set()
        for point in list:
            set_of_points.add(tuple(point.tolist()))
        return set_of_points

    def insert(self, point: np.ndarray):
        """
        Insert a new point into the data structure.
        
        Converts the numpy array to a tuple and adds it to the set.
        Duplicate points are automatically ignored due to set properties.
        
        Time complexity: O(1)
        
        Args:
            point (np.ndarray): Point to insert
        """
        point = tuple(point.tolist())
        if point not in self.points: 
            self.points.add(point)
    
    def get_knn(self, point: np.ndarray, k: int): 
        """
        Get k nearest neighbors to the target point.
        
        Uses a linear scan through all points and maintains a max-heap of the
        k closest points seen so far.
        
        Time complexity: O(n log k) where n is the number of points
        
        Args:
            point (np.ndarray): Target point
            k (int): Number of neighbors to retrieve
            
        Returns:
            List[List[float]]: k nearest points as lists, sorted by distance (closest first)
        """
        priority_queue = []  # Max-heap of (-distance, index, point)
        for index, curr_point in enumerate(self.points):
            curr_dist = self.dist_function(pointA=point, pointB=np.array(curr_point))
            if len(priority_queue) < k:
                heapq.heappush(priority_queue, (-curr_dist, index, curr_point))
            elif curr_dist < -priority_queue[0][0]:
                heapq.heappushpop(priority_queue, (-curr_dist, index, curr_point)) 

        # Sort results by distance (closest first) and convert to lists
        return [list(queue_element[2]) for queue_element in sorted(priority_queue, reverse=True)]
    
    def delete(self, point: np.ndarray): 
        """
        Delete a point from the data structure.
        
        Converts the numpy array to a tuple and removes it from the set.
        
        Time complexity: O(1)
        
        Args:
            point (np.ndarray): Point to delete
            
        Raises:
            KeyError: If the point is not in the data structure
        """
        point = tuple(point.tolist())
        self.points.remove(point)
    
    def get_nearest(self, point: np.ndarray): 
        """
        Get the single nearest neighbor to the target point.
        
        Uses a linear scan through all points to find the closest one.
        
        Time complexity: O(n) where n is the number of points
        
        Args:
            point (np.ndarray): Target point
            
        Returns:
            List[float]: The nearest point as a list
            
        Raises:
            ValueError: If the data structure is empty
        """
        if not self.points:
            raise ValueError("Cannot find nearest neighbor in an empty data structure")
            
        nearest_point = None
        nearest_dist = float("inf")
        for curr_point in self.points:
            curr_dist = self.dist_function(pointA=point, pointB=np.array(curr_point))
            if curr_dist < nearest_dist:
                nearest_point = curr_point
                nearest_dist = curr_dist
        return list(nearest_point)

    def query_range(self, center_point: np.ndarray, radius: float):
        """
        Get all points within the specified radius of the center point.
        
        Uses a linear scan through all points to identify those within the radius.
        Note that the comparison uses squared distance for efficiency.
        
        Time complexity: O(n) where n is the number of points
        
        Args:
            center_point (np.ndarray): The center of the search sphere
            radius (float): The search radius
            
        Returns:
            List[List[float]]: Points within the radius as lists
        """
        radius_square = radius**2  # Square the radius to compare with squared distances
        results_list = []
        for point in self.points:
            if self.dist_function(pointA=center_point, pointB=np.array(point)) <= radius_square:
                results_list.append(list(point))

        return results_list


