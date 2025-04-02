"""
Abstract base class for geometric data structures.

This module defines the interface that all geometric data structures must implement,
including operations for construction, querying, and modification. Concrete implementations
like KD-Tree, Ball*-Tree, and Brute Force search extend this base class.

Classes:
    GeometricDataStructure: Abstract base class for spatial data structures
"""

from typing import *
import os
import sys
import numpy as np

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree.utils import euclidean_distance
from tree.utils import euclidean_squared_distance

class GeometricDataStructure:
    """
    Abstract base class for geometric data structures.
    
    This class defines the interface that must be implemented by all geometric
    data structures used for spatial queries (nearest neighbor, range search, etc.).
    Concrete subclasses must implement the abstract methods.
    
    Attributes:
        points (List[np.ndarray]): The dataset points stored in the structure.
        dimension (int): The dimensionality of the points.
        dist_function (Callable): Distance function used for comparing points.
    """
    
    def __init__(self,
                 dimension: int,
                 points: Optional[List[np.ndarray]] = None,
                 dist_function: Optional[Callable] = None 
                ):
        """
        Initialize the geometric data structure.
        
        Args:
            dimension (int): The dimensionality of the points.
            points (Optional[List[np.ndarray]]): Initial dataset points, or None for empty structure.
            dist_function (Optional[Callable]): Custom distance function, or None to use Euclidean squared distance.
        """
        self.points = points
        self.dimension = dimension
        self.dist_function = euclidean_squared_distance if dist_function is None else dist_function

    def insert(self, point: np.ndarray):
        """
        Insert a new point into the data structure.
        
        Args:
            point (np.ndarray): The point to insert, must have dimension matching self.dimension.
            
        Raises:
            Exception: Abstract method that must be implemented by subclasses.
        """
        raise Exception("This function needs to be defined in subclass")
    
    def get_knn(self, point: np.ndarray, k: int):
        """
        Find the k nearest neighbors to the given point.
        
        Args:
            point (np.ndarray): The query point.
            k (int): Number of nearest neighbors to find.
            
        Returns:
            List[np.ndarray]: List of k nearest points, sorted by distance.
            
        Raises:
            Exception: Abstract method that must be implemented by subclasses.
        """
        raise Exception("This function needs to be defined in subclass")
    
    def delete(self, point: np.ndarray):
        """
        Delete a point from the data structure.
        
        Args:
            point (np.ndarray): The point to delete.
            
        Raises:
            Exception: Abstract method that must be implemented by subclasses.
        """
        raise Exception("This function needs to be defined in subclass")
    
    def get_nearest(self, point: np.ndarray):
        """
        Find the single nearest neighbor to the given point.
        
        Args:
            point (np.ndarray): The query point.
            
        Returns:
            np.ndarray: The nearest point.
            
        Raises:
            Exception: Abstract method that must be implemented by subclasses.
        """
        raise Exception("This function needs to be defined in subclass")
    
    def query_range(self, center_point: np.ndarray, radius: float):
        """
        Find all points within the specified radius of the center point.
        
        Args:
            center_point (np.ndarray): The center of the search sphere.
            radius (float): The radius of the search sphere.
            
        Returns:
            List[np.ndarray]: List of points within the specified radius.
            
        Raises:
            Exception: Abstract method that must be implemented by subclasses.
        """
        raise Exception("This function needs to be defined in subclass")
        
    def __sizeof__(self):
        """
        Calculate the approximate memory usage of this data structure.
        
        Returns:
            int: The total memory usage in bytes.
            
        Note:
            Subclasses should override this method to provide accurate memory usage estimates.
        """
        return sys.getsizeof(self.__dict__)

