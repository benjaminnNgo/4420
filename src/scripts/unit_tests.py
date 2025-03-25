import os
import sys
import unittest
import numpy as np

# Adjust the path to import the tree module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import KDTree, BallTree, BruteForce

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

    def test_range_search_kd_tree(self):
        """Test that KDTree returns the correct points within a radius"""
        query = np.array([5, 5])
        radius = 2.5
        
        # Get points within radius
        result = self.kd_tree.query_range(query, radius)
        
        # Expected points within radius 2.5 of [5,5]:
        # [3,5], [4,6], [5,6], [5,8]
        expected_set = arrays_to_set([np.array(p) for p in [
            [3, 5], [4, 6], [5, 6], [3, 4], [6, 7]
        ]])
        
        # Convert results to sets for comparison (ignoring order)
        result_set = arrays_to_set(result)
        
        self.assertEqual(result_set, expected_set,
                        f"KDTree range search should return correct points within radius. Got {result_set}, expected {expected_set}")

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
        
    def test_range_search_ball_tree(self):
        """Test that BallTree returns the correct points within a radius"""
        query = np.array([5, 5])
        radius = 2.5
        
        # Get points within radius
        result = self.ball_tree.query_range(query, radius)
        
        # Convert results to sets for comparison (ignoring order)
        result_set = arrays_to_set(result)
        expected_set = arrays_to_set([np.array(p) for p in [
            [3, 5], [4, 6], [5, 6], [3, 4], [6, 7]
        ]])
        
        self.assertEqual(result_set, expected_set,
                        f"BallTree range search should return correct points within radius. Got {result_set}, expected {expected_set}")

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
        
    def test_range_search_comparison(self):
        """Test that all three implementations return the same points for range search"""
        # Test with a few different queries and radii
        test_cases = [
            (np.array([5, 5]), 2.5),  # Medium radius
            (np.array([3, 3]), 1.0),  # Small radius
            (np.array([9, 9]), 10.0), # Large radius covering all points
            (np.array([20, 20]), 1.0) # No points in range
        ]
        
        for query, radius in test_cases:
            kd_result = self.kd_tree.query_range(query, radius)
            ball_result = self.ball_tree.query_range(query, radius)
            brute_result = self.brute.query_range(query, radius)
            
            # Convert to sets for comparison
            kd_set = arrays_to_set(kd_result)
            ball_set = arrays_to_set(ball_result)
            brute_set = arrays_to_set(brute_result)
            
            self.assertEqual(kd_set, brute_set,
                            f"KDTree and BruteForce range search results should match for query {query} and radius {radius}")
            self.assertEqual(ball_set, brute_set,
                            f"BallTree and BruteForce range search results should match for query {query} and radius {radius}")
            
    def test_sizeof_method(self):
        """Test and print the memory size of each data structure"""
        # Get initial sizes
        kd_size = self.kd_tree.__sizeof__()
        ball_size = self.ball_tree.__sizeof__()
        brute_size = self.brute.__sizeof__()
        
        print("\n===== Memory Usage Test =====")
        print(f"Initial KDTree size: {kd_size} bytes")
        print(f"Initial BallTree size: {ball_size} bytes")
        print(f"Initial BruteForce size: {brute_size} bytes")
        
        # Verify sizes are reasonable
        self.assertGreater(kd_size, 0, "KDTree size should be positive")
        self.assertGreater(ball_size, 0, "BallTree size should be positive")
        self.assertGreater(brute_size, 0, "BruteForce size should be positive")
        
        # Add more points to see if size increases
        additional_points = [np.array(p) for p in [
            [10, 10], [11, 11], [12, 12], [13, 13], [14, 14]
        ]]
        
        for point in additional_points:
            self.kd_tree.insert(point)
            self.ball_tree.insert(point)
            self.brute.insert(point)
        
        # Get new sizes
        kd_size_after = self.kd_tree.__sizeof__()
        ball_size_after = self.ball_tree.__sizeof__()
        brute_size_after = self.brute.__sizeof__()
        
        print(f"KDTree size after adding 5 points: {kd_size_after} bytes (Δ: {kd_size_after - kd_size})")
        print(f"BallTree size after adding 5 points: {ball_size_after} bytes (Δ: {ball_size_after - ball_size})")
        print(f"BruteForce size after adding 5 points: {brute_size_after} bytes (Δ: {brute_size_after - brute_size})")
        
        # Verify sizes increased
        self.assertGreater(kd_size_after, kd_size, "KDTree size should increase after adding points")
        self.assertGreater(ball_size_after, ball_size, "BallTree size should increase after adding points")
        self.assertGreater(brute_size_after, brute_size, "BruteForce size should increase after adding points")
        print("===== Memory Usage Test Complete =====\n")

if __name__ == '__main__':
    unittest.main()