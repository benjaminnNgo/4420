import os
import sys
import unittest
import numpy as np

# Adjust the path to import the tree module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree, BallTree, BruteForce
from data_loader.data_utils import data_loader

def make_hashable(item):
    if isinstance(item, np.ndarray):
        return tuple(item.tolist())
    elif isinstance(item, (list, tuple)):
        return tuple(make_hashable(subitem) for subitem in item)
    else:
        return item

def arrays_to_set(arrays):
    # Convert each element (and its sub-elements) to a hashable tuple  
    return set(make_hashable(arr) for arr in arrays)

class TestKDTree(unittest.TestCase):
    def setUp(self):
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        self.kd_tree = KDTree(dimension=2, points=self.points)

    def test_get_knn(self):
        query = np.array([9, 9])
        k = 2
        result = self.kd_tree.get_knn(query, k)
        self.assertEqual(len(result), k,
                         "KDTree.get_knn should return exactly k neighbors")
        # Expected nearest neighbors computed with Euclidean squared distance.
        expected_neighbors = [np.array([8, 9]), np.array([7, 8])]
        for expected in expected_neighbors:
            self.assertTrue(any(np.array_equal(expected, neighbor) for neighbor in result),
                            f"Expected neighbor {expected} not found in result {result}")

    def test_delete_insert(self):
        target = np.array([3, 4])
        # Ensure the target is initially in the tree:
        kd_nearest_before = self.kd_tree.get_nearest(target)
        self.assertTrue(np.array_equal(kd_nearest_before, target),
                        "Before deletion, the nearest neighbor for target should be itself")
        # Delete the target:
        self.kd_tree.delete(target)
        kd_nearest_after = self.kd_tree.get_nearest(target)
        self.assertFalse(np.array_equal(kd_nearest_after, target),
                         "After deletion, the target should not be returned as the nearest neighbor")
        # Reinsert the target:
        self.kd_tree.insert(target)
        kd_nearest_reinsert = self.kd_tree.get_nearest(target)
        self.assertTrue(np.array_equal(kd_nearest_reinsert, target),
                        "After reinsertion, the target should again be returned as the nearest neighbor")

class TestBallTree(unittest.TestCase):
    def setUp(self):
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        # Copy the list to avoid side-effects between tests.
        self.ball_tree = BallTree(dimension=2, points=self.points.copy())

    def test_delete_insert(self):
        target = np.array([3, 4])
        # Ensure the target is initially in the tree:
        ball_nearest_before = self.ball_tree.get_nearest(target)
        self.assertTrue(np.array_equal(ball_nearest_before, target),
                        "Before deletion, the nearest neighbor for target should be itself")
        # Delete the target:
        self.ball_tree.delete(target)
        ball_nearest_after = self.ball_tree.get_nearest(target)
        self.assertFalse(np.array_equal(ball_nearest_after, target),
                         "After deletion, the target should not be returned as the nearest neighbor")
        # Reinsert the target:
        self.ball_tree.insert(target)
        ball_nearest_reinsert = self.ball_tree.get_nearest(target)
        self.assertTrue(np.array_equal(ball_nearest_reinsert, target),
                        "After reinsertion, the target should again be returned as the nearest neighbor")

    def test_get_nearest_and_knn(self):
        query = np.array([9, 9])
        nearest = self.ball_tree.get_nearest(query)
        # Assuming [8, 9] is the nearest point to [9, 9]
        self.assertTrue(np.array_equal(nearest, np.array([8, 9])),
                        "Nearest point should be [8, 9], not " + str(nearest))
        k = 2
        knn = self.ball_tree.get_knn(query, k)
        self.assertEqual(len(knn), k,
                         "BallTree.get_knn should return exactly k neighbors")

class TestTreeComparison(unittest.TestCase):
    def setUp(self):
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        self.query = np.array([9, 9])
        self.k = 2
        self.kd_tree = KDTree(dimension=2, points=self.points.copy())
        self.ball_tree = BallTree(dimension=2, points=self.points.copy())
        self.brute = BruteForce(dimension=2, points=self.points.copy())

    def test_get_knn_comparison(self):
        # Get neighbors using each method
        kd_neighbors = self.kd_tree.get_knn(self.query, self.k)
        ball_neighbors = self.ball_tree.get_knn(self.query, self.k)
        brute_neighbors = self.brute.get_knn(self.query, self.k)
        
        kd_set = arrays_to_set(kd_neighbors)
        ball_set = arrays_to_set(ball_neighbors)
        brute_set = arrays_to_set(brute_neighbors)

        self.assertEqual(kd_set, brute_set,
                        "KDTree.get_knn output should match BruteForce.get_knn output")
        self.assertEqual(ball_set, brute_set,
                        "BallTree.get_knn output should match BruteForce.get_knn output. Bruteforce: " + str(brute_set) + " Ball: " + str(ball_set))

    def test_get_nearest_comparison(self):
        kd_nearest = self.kd_tree.get_nearest(self.query)
        ball_nearest = self.ball_tree.get_nearest(self.query)
        brute_nearest = self.brute.get_nearest(self.query)

        self.assertTrue(np.array_equal(kd_nearest, brute_nearest),
                        "KDTree.get_nearest output should match BruteForce.get_nearest")
        self.assertTrue(np.array_equal(ball_nearest, brute_nearest),
                        "BallTree.get_nearest output should match BruteForce.get_nearest")

if __name__ == '__main__':
    unittest.main()