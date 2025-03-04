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
Overview: Bruteforce implementation of geometric data structure
"""
        
class BruteForce (GeometricDataStructure):
    r"""Brute force implementation of geometric data structure.
    This serve as a baseline for comparision

    Args:
        dimension (int): dimension of each point
        points (List[List]) : list of initial points
        dist_function (function pointer) : function to compute distance between 2 points, by `default` is `euclidean_squ_distance`
    """
    def __init__(self, dimension : int,
                 points:Optional[List[List]] = None,
                 dist_function : Optional[Callable] = None):
        super().__init__(dimension,points, dist_function)
        
        self.points = self._to_set_of_tuple(self.points)
    
    def _to_set_of_tuple(self,
                         list: List[List])-> Set[Tuple]:
        r"""
        Private function: convert from list of point to set of point, where each
        point is represented as a tuple
        """
        set_of_points = set()
        for point in list:
            set_of_points.add(tuple(point))
        return set_of_points


    def insert(self,
               point:List[List]):
        r"""
        Insert a new node to the tree

        Args:
            point (List[List]) : new node
        """
        point = tuple(point)
        if point not in self.points: 
            self.points.add(point)
    
    def get_knn(self,
                point: List[List], 
                k:int): 
        r"""
        Get k nearest neighbours 

        Args:
            point (np.array) : target point 
            k (int) : indicate the number of neighbours to query 
        """
        priority_queue = []
        for index,curr_point in enumerate(self.points):
            curr_dist = self.dist_function(pointA= point, pointB= np.array(curr_point))
            if len(priority_queue) < k:
                heapq.heappush(priority_queue,(-curr_dist,index,curr_point))
            elif curr_dist < -priority_queue[0][0]:
                heapq.heappushpop(priority_queue,(-curr_dist,index,curr_point)) 


        return [ list(queue_element[2]) for queue_element in sorted(priority_queue, reverse= True)]

    
    def delete(self,
               point : List[List]): 
        r"""
        Delete an element from the data structure

        Args:
            point (List[List]) : new node
        """
        point = tuple(point)
        self.points.remove(point)
    
    def get_nearest(self,
                    point : List[List]): 
        r"""
        Get the nearest neighbours 

        Args:
            point (np.array) : target point 
        """
        point = np.array(point)
        nearest_point = None
        nearest_dist = float("inf")
        for curr_point in self.points:
            curr_dist = self.dist_function(pointA= point, pointB= np.array(curr_point))
            if curr_dist < nearest_dist:
                nearest_point = curr_point
                nearest_dist = curr_dist
        return list(nearest_point)

    def query_range(self,
                    center_point: List[List], 
                    radius:int):
        r"""
        Get all points that having distance with center point within a given range

        Args:
            center_point (List[List]): target point
            radius (int) : range
        """
        radius_square = radius**2
        center_point = np.array(center_point)
        results_list = []
        for point in self.points:
            if self.dist_function(pointA=center_point,pointB=np.array(point)) < radius_square:
                results_list.append(list(point))

        return results_list

