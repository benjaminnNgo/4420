import os
import sys
import numpy as np
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree
from tree.utils import euclidean_distance

#[1,2],[2,3],[3,4],[5,6],[5,3]]
test_tree = KDTree(2)
test_tree.insert([1,2])
test_tree.insert([2,3])
test_tree.insert([0,3])
test_tree.insert([3,5])



print(test_tree.root.coordinate)
print(test_tree.root.right.right.coordinate)


# print(euclidean_distance(np.array([1,2]), np.array([3,4])))