import os
import sys
import numpy as np
# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree, BruteForce
from tree.utils import euclidean_squ_distance



# print(euclidean_squ_distance(np.array([1,2]), np.array([3,4])))

print((type(np.array([3,4]))))
# [1,2],[2,3],[3,4],[5,6],[5,3]]
points = [
        [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
        [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
    ]

for i in range(len(points)):
    points[i] = np.array(points[i])
test_tree = KDTree(dimension=2, points=points)
test_baseline = BruteForce(dimension=2, points=points)
result = test_tree.get_knn(np.array([9, 9]),2)
print(result)
print(test_baseline.points)
test_baseline.delete(np.array([3,4]))
print(test_baseline.points)
test_baseline.insert(np.array([3,4]))
print(test_baseline.points)

print(test_baseline.get_nearest(np.array([9, 9])))
print(test_baseline.get_knn(np.array([9, 9]),2))



# print(test_tree.root.coordinate)
# print(test_tree.root.right.right.coordinate)


# print(euclidean_distance(np.array([1,2]), np.array([3,4])))