import os
import sys
import numpy as np
from typing import Optional
from typing import *

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure


        
class BruteForce (GeometricDataStructure):
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