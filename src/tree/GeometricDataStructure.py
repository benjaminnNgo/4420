from typing import *
import os
import sys
import numpy as np

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree.utils import euclidean_distance
from tree.utils import euclidean_squared_distance

"""
Authors: Tran Gia Bao Ngo
Overview: Base class for geometric data structure
"""

class GeometricDataStructure:
    def __init__(self,
                 dimension : int,
                 points:Optional[List[np.ndarray]] = None,
                 dist_function : Optional[Callable] = None 
                 ):
        self.points = points
        self.dimension = dimension
        self.dist_function = euclidean_squared_distance if dist_function is None else dist_function

    def insert(self,point:np.ndarray): 
        raise Exception("This function need to be defined in subclass")
    
    def get_knn(self,point: np.ndarray, k:int): 
        raise Exception("This function need to be defined in subclass")
    
    def delete(self,point : np.ndarray): 
        raise Exception("This function need to be defined in subclass")
    
    def get_nearest(self,point : np.ndarray): 
        raise Exception("This function need to be defined in subclass")
    
    def query_range(self,center_point: np.ndarray, radius:float):
        raise Exception("This function need to be defined in subclass")
    
