import os
import sys
import numpy as np
from typing import Optional
from typing import *

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure

class KDTreeNode:
    def __init__(self,coordinate:np.array,
                 compare_axis: int, 
                 left: Optional["KDTreeNode"] = None,
                 right: Optional["KDTreeNode"] = None):
        self.coordinate = coordinate
        self.compare_axis = compare_axis
        self.right = right
        self.left = left
        
class KDTree (GeometricDataStructure):
    def __init__(self, points, dimension, dist_function = None):
        super().__init__(points, dimension, dist_function)
        self.root = self._construct_tree(points=points, depth=0)


    def _construct_tree(self,points: List[List], depth = 0):
        if not points:
            return None
        
        compared_axis = depth % self.dimension
        sorted_points = sorted(points, key= lambda point: point[compared_axis])
        median_point_idx = len(sorted_points)//2
        new_node = KDTreeNode(coordinate= np.array(sorted_points[median_point_idx]), compare_axis= compared_axis)
        new_node.left = self._construct_tree(points= sorted_points[:median_point_idx], depth= depth+1)
        new_node.right = self._construct_tree(points= sorted_points[median_point_idx+1:], depth= depth+1)
        return new_node
        

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
    
   