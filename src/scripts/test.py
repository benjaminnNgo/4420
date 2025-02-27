import os
import sys
import numpy as np
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree
from tree.utils import euclidean_distance

test_tree = KDTree([[1,2],[2,3],[3,4],[5,6],[5,3]],2)
print(test_tree.root.right.coordinate)

print(euclidean_distance(np.array([1,2]), np.array([3,4])))