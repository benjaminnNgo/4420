import os
import sys
import unittest
import numpy as np

# Adjust the path to import from the tree module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree, BallTree, BruteForce

def make_hashable(item):
    """
    Convert an item (such as a numpy array) into a hashable tuple.
    
    This is required because sets (and dictionary keys) need hashable items.
    """
    if isinstance(item, np.ndarray):
        return tuple(item.tolist())
    elif isinstance(item, (list, tuple)):
        return tuple(make_hashable(subitem) for subitem in item)
    else:
        return item

def arrays_to_set(arrays):
    """
    Convert a collection of arrays into a set of hashable tuples.
    
    This facilitates comparison of lists of numpy arrays without worrying about their order.
    """
    return set(make_hashable(arr) for arr in arrays)

class TestKDTree(unittest.TestCase):
    def setUp(self):
        # Initialize a sample dataset of 2D points for KDTree tests.
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        self.kd_tree = KDTree(dimension=2, points=self.points)

    def test_get_knn(self):
        # Test k-nearest neighbors (k-NN) query.
        query = np.array([9, 9])
        k = 2
        result = self.kd_tree.get_knn(query, k)
        # Check that exactly k neighbors are returned.
        self.assertEqual(len(result), k,
                         "KDTree.get_knn should return exactly k neighbors")
        # Expected nearest neighbors computed with Euclidean squared distance.
        expected_neighbors = [np.array([8, 9]), np.array([7, 8])]
        # Verify that each expected neighbor exists in the result.
        for expected in expected_neighbors:
            self.assertTrue(any(np.array_equal(expected, neighbor) for neighbor in result),
                            f"Expected neighbor {expected} not found in result {result}")

    def test_delete_insert(self):
        # Test deletion and subsequent reinsertion of a point.
        target = np.array([3, 4])
        
        # Verify that the target is initially present and is returned as its own nearest neighbor.
        kd_nearest_before = self.kd_tree.get_nearest(target)
        self.assertTrue(np.array_equal(kd_nearest_before, target),
                        "Before deletion, the nearest neighbor for target should be itself")
        
        # Delete the target and test that it is no longer returned as nearest.
        self.kd_tree.delete(target)
        kd_nearest_after = self.kd_tree.get_nearest(target)
        self.assertFalse(np.array_equal(kd_nearest_after, target),
                         "After deletion, the target should not be returned as the nearest neighbor")
        
        # Reinsert the target and verify that it once again becomes the nearest neighbor.
        self.kd_tree.insert(target)
        kd_nearest_reinsert = self.kd_tree.get_nearest(target)
        self.assertTrue(np.array_equal(kd_nearest_reinsert, target),
                        "After reinsertion, the target should again be returned as the nearest neighbor")

class TestBallTree(unittest.TestCase):
    def setUp(self):
        # Initialize a sample dataset of 2D points for BallTree tests.
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        # Copy the list to avoid side-effects between tests.
        self.ball_tree = BallTree(dimension=2, points=self.points.copy())

    def test_delete_insert(self):
        # Test deletion and reinsertion in BallTree.
        target = np.array([3, 4])
        
        # Verify that the target exists initially.
        ball_nearest_before = self.ball_tree.get_nearest(target)
        self.assertTrue(np.array_equal(ball_nearest_before, target),
                        "Before deletion, the nearest neighbor for target should be itself")
        
        # Delete the target and verify that it is not found.
        self.ball_tree.delete(target)
        ball_nearest_after = self.ball_tree.get_nearest(target)
        self.assertFalse(np.array_equal(ball_nearest_after, target),
                         "After deletion, the target should not be returned as the nearest neighbor")
        
        # Reinsert the target and ensure it becomes the nearest neighbor again.
        self.ball_tree.insert(target)
        ball_nearest_reinsert = self.ball_tree.get_nearest(target)
        self.assertTrue(np.array_equal(ball_nearest_reinsert, target),
                        "After reinsertion, the target should again be returned as the nearest neighbor")

    def test_get_nearest_and_knn(self):
        # Test both get_nearest and get_knn functions.
        query = np.array([9, 9])
        nearest = self.ball_tree.get_nearest(query)
        
        # Assuming [8, 9] is the nearest point to the query.
        self.assertTrue(np.array_equal(nearest, np.array([8, 9])),
                        "Nearest point should be [8, 9], not " + str(nearest))
        
        k = 2
        knn = self.ball_tree.get_knn(query, k)
        self.assertEqual(len(knn), k,
                         "BallTree.get_knn should return exactly k neighbors")
        

class TestTreeComparison(unittest.TestCase):
    def setUp(self):
        # Set up the same dataset for cross-comparison between KDTree, BallTree, and BruteForce.
        self.points = [np.array(p) for p in [
            [1, 2], [3, 4], [5, 6], [7, 8], [2, 3],
            [6, 7], [8, 9], [3, 5], [4, 6], [5, 8]
        ]]
        # Define the query and number of neighbors.
        self.query = np.array([9, 9])
        self.k = 2
        
        # Initialize all three data structures with the same set of points.
        self.kd_tree = KDTree(dimension=2, points=self.points.copy())
        self.ball_tree = BallTree(dimension=2, points=self.points.copy())
        self.brute = BruteForce(dimension=2, points=self.points.copy())

    def test_get_knn_comparison(self):
        # Compare k-NN outputs across the three different implementations.
        kd_neighbors = self.kd_tree.get_knn(self.query, self.k)
        ball_neighbors = self.ball_tree.get_knn(self.query, self.k)
        brute_neighbors = self.brute.get_knn(self.query, self.k)
        
        kd_set = arrays_to_set(kd_neighbors)
        ball_set = arrays_to_set(ball_neighbors)
        brute_set = arrays_to_set(brute_neighbors)

        # Ensure that KDTree and BallTree outputs match the BruteForce result.
        self.assertEqual(kd_set, brute_set,
                         "KDTree.get_knn output should match BruteForce.get_knn output")
        self.assertEqual(ball_set, brute_set,
                         "BallTree.get_knn output should match BruteForce.get_knn output. "
                         "BruteForce: " + str(brute_set) + " BallTree: " + str(ball_set))

    def test_get_nearest_comparison(self):
        # Compare the single nearest neighbor search.
        kd_nearest = self.kd_tree.get_nearest(self.query)
        ball_nearest = self.ball_tree.get_nearest(self.query)
        brute_nearest = self.brute.get_nearest(self.query)

        self.assertTrue(np.array_equal(kd_nearest, brute_nearest),
                         "KDTree.get_nearest output should match BruteForce.get_nearest")
        self.assertTrue(np.array_equal(ball_nearest, brute_nearest),
                         "BallTree.get_nearest output should match BruteForce.get_nearest")
            
    def test_sizeof_method(self):
        # Test the memory usage reporting of each data structure.
        kd_size = self.kd_tree.__sizeof__()
        ball_size = self.ball_tree.__sizeof__()
        brute_size = self.brute.__sizeof__()
        
        print("\n===== Memory Usage Test =====")
        print(f"Initial KDTree size: {kd_size} bytes")
        print(f"Initial BallTree size: {ball_size} bytes")
        print(f"Initial BruteForce size: {brute_size} bytes")
        
        # Verify that initial memory sizes are positive.
        self.assertGreater(kd_size, 0, "KDTree size should be positive")
        self.assertGreater(ball_size, 0, "BallTree size should be positive")
        self.assertGreater(brute_size, 0, "BruteForce size should be positive")
        
        # Insert additional points and check that the memory usage increases.
        additional_points = [np.array(p) for p in [
            [10, 10], [11, 11], [12, 12], [13, 13], [14, 14]
        ]]
        
        for point in additional_points:
            self.kd_tree.insert(point)
            self.ball_tree.insert(point)
            self.brute.insert(point)
        
        kd_size_after = self.kd_tree.__sizeof__()
        ball_size_after = self.ball_tree.__sizeof__()
        brute_size_after = self.brute.__sizeof__()
        
        print(f"KDTree size after adding 5 points: {kd_size_after} bytes (Δ: {kd_size_after - kd_size})")
        print(f"BallTree size after adding 5 points: {ball_size_after} bytes (Δ: {ball_size_after - ball_size})")
        print(f"BruteForce size after adding 5 points: {brute_size_after} bytes (Δ: {brute_size_after - brute_size})")
        print("===== Memory Usage Test Complete =====\n")
        
        # Assert that the sizes increased after inserting new points.
        self.assertGreater(kd_size_after, kd_size, "KDTree size should increase after adding points")
        self.assertGreater(ball_size_after, ball_size, "BallTree size should increase after adding points")
        self.assertGreater(brute_size_after, brute_size, "BruteForce size should increase after adding points")

if __name__ == '__main__':
    unittest.main()