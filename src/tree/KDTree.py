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
Overview: Implementation of KD-tree from the `"Multidimensional binary search trees used for associative searching" <https://doi.org/10.1145/361002.361007>
"""

class KDTreeNode:
    r"""Representing a node in KD-tree

    Args:
        coordinate (np.array): 1D vector recording coordinates of a point
        left (KDTreeNode) : left child
        right (KDTreeNode) : right child
    """

    def __init__(self,
                 coordinate:np.array,
                 left: Optional["KDTreeNode"] = None,
                 right: Optional["KDTreeNode"] = None
                ):
        
        self.coordinate = coordinate
        self.right = right
        self.left = left
        
class KDTree(GeometricDataStructure):
    r"""KD-tree from the `"Multidimensional binary search trees used for associative searching" <https://doi.org/10.1145/361002.361007>`

    Args:
        dimension (int): dimension of each point
        points (List[np.ndarray]) : list of initial points
        dist_function (function pointer) : function to compute distance between 2 points, by `default` is `euclidean_squ_distance`
    """
    def __init__(self,
                 dimension : int,
                 points:Optional[List[np.ndarray]] = None,
                 dist_function : Optional[Callable] = None 
                ):
        super().__init__(dimension,points,dist_function)
        self.root = self._construct_tree(points=points, depth=0)


    def _construct_tree(self,
                        points: List[np.ndarray], 
                        depth = 0):
        r"""
        Private function: constructing an tree from scratch given list of points recursively

        Args:
            points (List[np.ndarray]) : list of points
            depth (int) : indicate current depth level when call this function. This is used to define which axis to compare
        """
        if not points:
            return None
        
        compared_axis = depth % self.dimension #define which axis used to split tree into left and right branches
        sorted_points = sorted(points, key= lambda point: point[compared_axis])
        median_point_idx = len(sorted_points)//2 #Get the median node to split list of nodes into 2 approximately equal 2 sublist
        
        new_node = KDTreeNode(coordinate= sorted_points[median_point_idx])
        new_node.left = self._construct_tree(points= sorted_points[:median_point_idx], depth= depth+1)
        new_node.right = self._construct_tree(points= sorted_points[median_point_idx+1:], depth= depth+1)
        return new_node
        
    def _insert(self, 
                root:KDTreeNode, 
                point:np.ndarray,
                depth:int = 0):
        r"""
        Private function: Given current node in a tree and a new point, decide whether to insert into left branch or right branch (recursively)
        If left/right child of current node is None (base case), insert directly 

        Args:
            root (KDTreeNode) : current node in the tree
            point (np.ndarray) : inserting point
            depth (int) : indicate current depth level when call this function. This is used to define which axis to compare
        """
        compared_axis = depth % self.dimension
        dx =  point[compared_axis] - root.coordinate[compared_axis]
        if dx > 0:
            if root.right is None:
                root.right =  KDTreeNode(coordinate= point)
            else:
                self._insert(root= root.right,point=point,depth=depth+1)
        else:

            if root.left is None:
                root.left =  KDTreeNode(coordinate= point)
            else:
                self._insert(root= root.left,point=point,depth=depth+1)



    def insert(self,
               point:np.ndarray): 
        r"""
        Insert a new node to the tree

        Args:
            point (np.ndarray) : new node
        """
        
        if self.root is None:
            self.root = KDTreeNode(coordinate= point, compare_axis= 0)
        else:
            self._insert(root= self.root, point = point,depth = 0)

    def delete(self,
               point : np.ndarray): 
        # @TODO: If delete leaf node, it is easy. But if delete internal node, we may need to re-build the enture right subtree
        raise Exception("This function need to be defined in subclass")


    def _get_knn(self, 
                 point:np.ndarray, 
                 compared_node: KDTreeNode, 
                 k: int, 
                 priority_queue: List, 
                 depth: int = 0, 
                 tiebreaker = 1):
        r"""
        Private function: Get k nearest neighbours recursively 

        Args:
            point (np.ndarray) : target point 
            compared_node (KDTreeNode) : current node in the tree
            k (int) : indicate the number of neighbours to query 
            priority_queue (List): priority queue to keep track of k nearest neighbours 
            depth (int) : depth of current node in the tree. This is used to define which axis to compare 
            tiebreaker (int): When 2 points having the same distance to target points, `heapq` will use this value to
                              define priority. In addition,`tiebreaker` for each element is unique. All of these to 
                              avoid randomness made by `heapq` in return results.
        """
        
        if compared_node is None: # Reach the end of the tree
            return
        
        compared_axis = depth % self.dimension
        curr_dist = self.dist_function(pointA= point, pointB= compared_node.coordinate)

        if len(priority_queue) < k:
            heapq.heappush(priority_queue,(-curr_dist,tiebreaker,compared_node))
        elif curr_dist < -priority_queue[0][0]:
            heapq.heappushpop(priority_queue,(-curr_dist,tiebreaker,compared_node)) 
            """
            The above line of code is equivalent to:
            heapq.heappop(priority_queue)
            heapq.heappush(priority_queue,(-square_dist,tiebreaker,compared_node)) 

            but for efficiency, we call heappushpop, which accomplish the same thing but faster
            """
            
        
        if point[compared_axis] < compared_node.coordinate[compared_axis]:
            # We prioritize go the left child in this case
            next_node = compared_node.left
            other_node = compared_node.right
        else:
            # We prioritize go the right child in this case
            next_node = compared_node.right
            other_node = compared_node.left
        
        self._get_knn(point=point, compared_node= next_node, k= k, priority_queue = priority_queue, depth = depth + 1, tiebreaker= 2*tiebreaker)
        if (point[compared_axis] - compared_node.coordinate[compared_axis])**2 < -priority_queue[0][0]:
            # We need to traverse other branch since we can find a closer node potentially in this case
            self._get_knn(point=point, compared_node= other_node, k= k, priority_queue = priority_queue, depth = depth + 1, tiebreaker= 2*tiebreaker + 1)

        if tiebreaker == 1:
            return [queue_element[2].coordinate.tolist() for queue_element in sorted(priority_queue, reverse= True)]
    
    def get_knn(self,
                point: np.ndarray,
                k:int): 
        r"""
        Get k nearest neighbours 

        Args:
            point (np.ndarray) : target point 
            k (int) : indicate the number of neighbours to query 
        """
        return self._get_knn(point = point, compared_node = self.root, k= k, priority_queue = [])    
    
    def get_nearest(self,
                    point : np.ndarray): 
        r"""
        Get the nearest neighbours 

        Args:
            point (np.ndarray) : target point 
        """
        return self._get_knn(point = np.array(point), compared_node = self.root, k= 1, priority_queue = [])[0]
    
    def query_range(self,
                    center_point: np.ndarray, 
                    radius:int):
        # is it the same way with self._get_knn() in the way it traverse the tree. Instead of keeping a priority queue, keeping a list of qualified points
        raise Exception("This function need to be defined in subclass")
    
   