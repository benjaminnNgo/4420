import os
import sys
import numpy as np
from typing import Optional
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure

class KDTreeNode:
    def __init__(self,coordinate:np.ndarray
                 ,compare_axis: int, 
                 left: Optional["KDTreeNode"] = None,
                 right: Optional["KDTreeNode"] = None):
        self.coordinate = coordinate
        self.compare_axis = compare_axis
        self.right = right
        self.left = left
        
class KDTreee (GeometricDataStructure):
    def insert(): #@TODO: define parameters here
        pass
    
    def get_knn(): #@TODO: define parameters here
        pass
    
    def delete(): #@TODO: define parameters here
        pass
    
    def get_nearest(): #@TODO: define parameters here
        pass