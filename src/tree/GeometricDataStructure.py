from typing import *
import os
import sys

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree.utils import euclidean_distance

class GeometricDataStructure:
    def __init__(self,
                 points:List[List],
                 dimension : int,
                 dist_function : Optional[Callable] = None 
                 ):
        self.points = points
        self.dimension = dimension
        self.dist_function = euclidean_distance if dist_function is None else dist_function

    def insert(point:List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    def get_knn(point: List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    def delete(point : List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    def get_nearest(point : List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    def query_range(center_point: List[List], radius:int):
        raise Exception("This function need to be defined in subclass")
    
