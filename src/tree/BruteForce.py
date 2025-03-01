import os
import sys
import numpy as np
from typing import Optional
from typing import *

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure


        
class BruteForce (GeometricDataStructure):
    def __init__(self, points, dimension, dist_function = None):
        super().__init__(points, dimension, dist_function)
        
        self.points = self._to_set_of_tuple(self.points)
    
    def _to_set_of_tuple(self,list)-> Set[Tuple]:
        set_of_points = set()
        for point in list:
            set_of_points.add((point[0],point[1]))
        return set_of_points


    def insert(self,point:List[List]):
        point = (point[0],point[1])
        if point not in self.points: 
            self.points.add(point)
    
    def get_knn(self,point: List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    def delete(self,point : List[List]): 
        point = (point[0],point[1])
        self.points.remove(point)
    
    def get_nearest(self,point : List[List]) -> List[List]: 
        point = np.array(point)
        nearest_point = None
        nearest_dist = float("inf")
        for curr_point in self.points:
            curr_dist = self.dist_function(pointA= point, pointB= np.array(curr_point))
            if curr_dist < nearest_dist:
                nearest_point = curr_point
                nearest_dist = curr_dist
        return nearest_point

    def query_range(self,center_point: List[List], radius:int) -> List[List[List]]:
        radius_square = radius**2
        center_point = np.array(center_point)
        results_list = []
        for point in self.points:
            if self.dist_function(pointA=center_point,pointB=np.array(point)) < radius_square:
                results_list.append(point)

        return results_list

