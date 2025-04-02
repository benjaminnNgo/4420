'''
class TestGroundtruthQueries(unittest.TestCase):
    def setUp(self):
        # Load the dataset (e.g., "cifar") from data_loader.
        # This assumes that data_loader returns an object with attributes:
        #   - dim: the feature dimension,
        #   - points: list of dataset points,
        #   - queries: list of query points,
        #   - groundtruth_queries: list of lists of indices corresponding to true neighbors.
        self.data = data_loader("cifar")
        self.dataset = self.data.points
        self.queries = self.data.queries
        self.groundtruth = self.data.groundtruth_queries  # e.g., groundtruth indices for each query

        self.kd_tree = KDTree(dimension=self.data.dim, points=self.dataset)
        self.ball_tree = BallTree(dimension=self.data.dim, points=self.dataset)
        self.brute = BruteForce(dimension=self.data.dim, points=self.dataset)

    def test_nearest_neighbor_groundtruth(self):
        # Test using the first query.
        query = np.array(self.queries[0])
        # Assume groundtruth for nearest neighbor is the first index for this query.
        gt_index = self.groundtruth[0][0]
        expected = self.dataset[gt_index]

        kd_nn = self.kd_tree.get_nearest(query)
        ball_nn = self.ball_tree.get_nearest(query)
        brute_nn = self.brute.get_nearest(query)

        self.assertTrue(np.allclose(kd_nn, expected),
                        "KDTree nearest neighbor does not match groundtruth")
        self.assertTrue(np.allclose(ball_nn, expected),
                        "BallTree nearest neighbor does not match groundtruth")
        self.assertTrue(np.allclose(brute_nn, expected),
                        "BruteForce nearest neighbor does not match groundtruth")

    def test_knn_groundtruth(self):
        # Test kNN using the first query.
        query = np.array(self.queries[0])
        k = len(self.groundtruth[0])
        # Expected neighbors are the dataset points at the groundtruth indices.
        expected = [self.dataset[i] for i in self.groundtruth[0]]

        kd_knn = self.kd_tree.get_knn(query, k)
        ball_knn = self.ball_tree.get_knn(query, k)
        brute_knn = self.brute.get_knn(query, k)

        # Convert lists of numpy arrays into sets of tuples for comparison.
        def to_set(neighbors):
            return set(tuple(pt.tolist()) for pt in neighbors)

        expected_set = to_set(expected)
        kd_set = to_set(kd_knn)
        ball_set = to_set(ball_knn)
        brute_set = to_set(brute_knn)

        self.assertEqual(kd_set, expected_set,
                         "KDTree kNN does not match groundtruth")
        self.assertEqual(ball_set, expected_set,
                         "BallTree kNN does not match groundtruth")
        self.assertEqual(brute_set, expected_set,
                         "BruteForce kNN does not match groundtruth")
'''