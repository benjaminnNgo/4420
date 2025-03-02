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
                 right: Optional["KDTreeNode"] = None
                ):
        self.coordinate = coordinate
        self.compare_axis = compare_axis
        self.right = right
        self.left = left
        
class KDTree (GeometricDataStructure):
    def __init__(self,
                 dimension : int,
                 points:Optional[List[List]] = None,
                 dist_function : Optional[Callable] = None 
                ):
        super().__init__(dimension,points,dist_function)
        self.root = self._construct_tree(points=points, depth=0)


    def _construct_tree(self,
                        points: List[List], 
                        depth = 0
                        ):
        if not points:
            return None
        
        compared_axis = depth % self.dimension
        sorted_points = sorted(points, key= lambda point: point[compared_axis])
        median_point_idx = len(sorted_points)//2
        new_node = KDTreeNode(coordinate= np.array(sorted_points[median_point_idx]), compare_axis= compared_axis)
        new_node.left = self._construct_tree(points= sorted_points[:median_point_idx], depth= depth+1)
        new_node.right = self._construct_tree(points= sorted_points[median_point_idx+1:], depth= depth+1)
        return new_node
        
    def _insert(self, 
                root:KDTreeNode, 
                point:List[List],
                depth:int = 0
                ):
        compared_axis = depth % self.dimension
        dx =  point[compared_axis] - root.coordinate[compared_axis]
        if dx > 0:
            if root.right is None:
                root.right =  KDTreeNode(coordinate= np.array(point), compare_axis= (depth+1) % self.dimension)
            else:
                self._insert(root= root.right,point=point,depth=depth+1)
        else:

            if root.left is None:
                root.left =  KDTreeNode(coordinate= np.array(point), compare_axis= (depth+1) % self.dimension)
            else:
                self._insert(root= root.left,point=point,depth=depth+1)



    def insert(self,
               point:List[List]
               ): 
        if self.root is None:
            self.root = KDTreeNode(coordinate= np.array(point), compare_axis= 0)
        else:
            self._insert(root= self.root, point = point,depth = 0)

    def delete(self,point : List[List]): 
        # @TODO: If delete leaf node, it is easy. But if delete internal node, we may need to re-build the enture right subtree
        raise Exception("This function need to be defined in subclass")

    
    def get_knn(self,point: List[List]): 
        raise Exception("This function need to be defined in subclass")
    
    
    def get_nearest(self,point : List[List], k:int): 
        raise Exception("This function need to be defined in subclass")
    
    def query_range(self,center_point: List[List], radius:int):
        raise Exception("This function need to be defined in subclass")
    
   