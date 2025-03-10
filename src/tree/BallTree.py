import os
import sys
import numpy as np
from typing import Optional
from typing import *
import heapq

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure
from utils import euclidean_squared_distance 

"""
Authors: Jared Rost
Overview: Implementation of Ball-tree from the `"Multidimensional binary search trees used for associative searching" <https://doi.org/10.1145/361002.361007>
"""

class BallTreeNode:
    r"""Node for the Ball*-tree.

    Attributes:
        center (np.array): center of the ball.
        radius (float): radius of the ball (covers all points in the node).
        left (Optional[BallTreeNode]): left child.
        right (Optional[BallTreeNode]): right child.
        points (Optional[List[np.ndarray]]): stored points at leaf nodes.
    """
    def __init__(self,
                 center: np.array,
                 radius: float,
                 left: Optional["BallTreeNode"] = None,
                 right: Optional["BallTreeNode"] = None,
                 points: Optional[List[np.ndarray]] = None):
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points

class BallTree(GeometricDataStructure):
    r"""Ball*-tree for efficient nearest neighbor search.

    This implementation follows the Ball*-tree algorithm in:
      http://arxiv.org/pdf/1511.00628

    Args:
        dimension (int): dimension of each point.
        points (List[np.ndarray]): list of initial points.
        dist_function (Callable): function to compute squared distance between 2 points.
                                    Default is Euclidean squared distance.
    """
    def __init__(self,
                 dimension: int,
                 points: Optional[List[np.ndarray]] = None,
                 dist_function: Optional[Callable] = None):
        if dist_function is None:
            dist_function = euclidean_squared_distance
        self.dimension = dimension
        self.dist_function = dist_function
        super().__init__(dimension, points, dist_function)
        self.leaf_size = 1  # you can adjust this threshold if needed
        self.root = self._construct_tree(points=points, depth=0)
    
    def _construct_tree(self,
                        points: List[np.ndarray],
                        depth: int = 0) -> Optional[BallTreeNode]:
        r"""
        Recursively constructs a Ball*-tree using PCA-based splitting.
        
        - If the number of points is less than or equal to leaf_size, create a leaf node.
        - Otherwise, compute the center and estimate the principal axis (via PCA).
        - Project points onto the principal axis and split at the median of the projections.
        - If one partition is empty, fall back to an even split.
        - Compute the node's radius as the maximum distance from the center.

        Args:
            points (List[np.ndarray]): list of points.
            depth (int): current recursion depth.

        Returns:
            BallTreeNode: the constructed ball tree node.
        """
        if points is None or len(points) == 0:
            return None
        if len(points) <= self.leaf_size:
            center = np.mean(points, axis=0)
            radius = np.sqrt(max(self.dist_function(center, pt) for pt in points))
            return BallTreeNode(center=center, radius=radius, points=points)
        
        # Compute the center.
        center = np.mean(points, axis=0)
        pts_array = np.vstack(points)
        # In case of a single point, return a leaf.
        if pts_array.shape[0] <= 1:
            return BallTreeNode(center=center, radius=0.0, points=points)
        # Compute covariance and its principal eigenvector.
        cov_matrix = np.cov(pts_array.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        principal_idx = np.argmax(eigenvalues)
        principal_axis = eigenvectors[:, principal_idx]
        
        # Project points onto the principal axis.
        projections = [np.dot(pt - center, principal_axis) for pt in points]
        median_proj = np.median(projections)
        left_points = [pt for pt, proj in zip(points, projections) if proj <= median_proj]
        right_points = [pt for pt, proj in zip(points, projections) if proj > median_proj]
        
        # Fallback: if one partition is empty, split evenly.
        if len(left_points) == 0 or len(right_points) == 0:
            mid = len(points) // 2
            left_points = points[:mid]
            right_points = points[mid:]
        
        radius = np.sqrt(max(self.dist_function(center, pt) for pt in points))
        left_child = self._construct_tree(left_points, depth+1)
        right_child = self._construct_tree(right_points, depth+1)
        return BallTreeNode(center=center, radius=radius, left=left_child, right=right_child)
    
    def _insert(self, 
                node: BallTreeNode, 
                point: np.ndarray,
                depth: int = 0) -> BallTreeNode:
        r"""
        Recursively inserts a point into the Ball*-tree.
        
        If a leaf node is reached (node.points is not None), then:
          - If the leaf is not full, add the point and update the ball.
          - If the leaf is full, rebuild the leaf using _construct_tree on all points.
        
        For internal nodes, update the bounding ball if necessary and then choose the child 
        whose center is closest to the new point.
        """
        if node is None:
            return BallTreeNode(center=point, radius=0.0, points=[point])
        
        if node.points is not None:
            if len(node.points) < self.leaf_size:
                node.points.append(point)
                new_center = np.mean(node.points, axis=0)
                new_radius = np.sqrt(max(self.dist_function(new_center, pt) for pt in node.points))
                node.center = new_center
                node.radius = new_radius
                return node
            else:
                all_points = node.points + [point]
                return self._construct_tree(all_points, depth)
        
        # At an internal node, update the sphere if needed.
        current_dist = np.sqrt(self.dist_function(node.center, point))
        if current_dist > node.radius:
            node.radius = current_dist
        
        # Choose the child with a closer center.
        left_dist = np.sqrt(self.dist_function(node.left.center, point)) if node.left is not None else float('inf')
        right_dist = np.sqrt(self.dist_function(node.right.center, point)) if node.right is not None else float('inf')
        
        if left_dist <= right_dist:
            node.left = self._insert(node.left, point, depth+1)
        else:
            node.right = self._insert(node.right, point, depth+1)
        return node

    def insert(self, point: np.ndarray):
        r"""
        Inserts a new point into the Ball*-tree by updating from the root.
        """
        self.root = self._insert(self.root, point, depth=0)
    
    def _collect_points(self, node: BallTreeNode) -> List[np.ndarray]:
        """
        Recursively collects all points from the subtree rooted at `node`.
        """
        if node is None:
            return []
        if node.points is not None:
            return node.points
        return self._collect_points(node.left) + self._collect_points(node.right)
    
    def _delete(self, node: BallTreeNode, point: np.ndarray, depth: int = 0) -> Tuple[Optional[BallTreeNode], bool]:
        r"""
        Recursively deletes a point from the Ball*-tree.
        
        If deletion is successful, rebuilds the subtree from the remaining points.
        """
        if node is None:
            return None, False
        
        if node.points is not None:
            if any(np.array_equal(pt, point) for pt in node.points):
                node.points = [pt for pt in node.points if not np.array_equal(pt, point)]
                if not node.points:
                    return None, True
                node.center = np.mean(node.points, axis=0)
                node.radius = np.sqrt(max(self.dist_function(node.center, pt) for pt in node.points))
                return node, True
            return node, False

        left_deleted = False
        node.left, left_deleted = self._delete(node.left, point, depth+1)
        if not left_deleted:
            node.right, left_deleted = self._delete(node.right, point, depth+1)
        
        if left_deleted:
            all_points = self._collect_points(node)
            if all_points:
                new_node = self._construct_tree(all_points, depth)
                return new_node, True
            else:
                return None, True
        return node, False

    def delete(self, point: np.ndarray):
        r"""
        Deletes a point from the Ball*-tree, rebuilding affected subtrees.
        """
        self.root, deleted = self._delete(self.root, point, depth=0)
        if not deleted:
            raise Exception("Point not found in the Ball*-tree")
    
    def _get_knn(self, 
                 point: np.ndarray, 
                 node: BallTreeNode, 
                 k: int, 
                 priority_queue: List, 
                 depth: int = 0, 
                 tiebreaker: int = 1) -> Tuple[List, int]:
        r"""
        Recursively retrieves k nearest neighbors using a max-heap priority queue.
        
        Prunes branches where the lower bound (based on node sphere) exceeds the worst current candidate.
        """
        if node is None:
            return priority_queue, tiebreaker
        
        center_distance = np.sqrt(self.dist_function(node.center, point))
        lower_bound = max(0, center_distance - node.radius)
        if len(priority_queue) == k and lower_bound >= -priority_queue[0][0]:
            return priority_queue, tiebreaker
        
        if node.points is not None:
            for pt in node.points:
                d = np.sqrt(self.dist_function(pt, point))
                if len(priority_queue) < k:
                    heapq.heappush(priority_queue, (-d, tiebreaker, pt))
                    tiebreaker += 1
                else:
                    worst = -priority_queue[0][0]
                    if d < worst:
                        heapq.heappushpop(priority_queue, (-d, tiebreaker, pt))
                        tiebreaker += 1
            return priority_queue, tiebreaker
        
        children = []
        if node.left is not None:
            left_center_distance = np.sqrt(self.dist_function(node.left.center, point))
            left_lb = max(0, left_center_distance - node.left.radius)
            children.append((left_lb, node.left))
        if node.right is not None:
            right_center_distance = np.sqrt(self.dist_function(node.right.center, point))
            right_lb = max(0, right_center_distance - node.right.radius)
            children.append((right_lb, node.right))
        children.sort(key=lambda x: x[0])
        for lb, child in children:
            priority_queue, tiebreaker = self._get_knn(point, child, k, priority_queue, depth+1, tiebreaker)
        return priority_queue, tiebreaker

    def get_knn(self,
                point: np.ndarray,
                k: int):
        r"""
        Retrieves k nearest neighbors of a target point.
        
        Returns:
            List[Tuple[float, np.ndarray]]: Sorted list of (distance, point) tuples.
        """
        priority_queue, _ = self._get_knn(point, self.root, k, [])
        knn = [(-d, pt) for d, _, pt in priority_queue]
        knn.sort(key=lambda x: x[0])
        return knn

    def get_nearest(self,
                    point: np.ndarray):
        r"""
        Retrieves the nearest neighbor of a target point.
        """
        knn = self.get_knn(point, k=1)
        return knn[0] if knn else None

    def query_range(self,
                    center_point: np.ndarray, 
                    radius: int):
        r"""
        Retrieves all points within a given radius of center_point.
        """
        result = []
        def _query(node: BallTreeNode):
            if node is None:
                return
            center_dist = np.sqrt(self.dist_function(node.center, center_point))
            if center_dist - node.radius > radius:
                return
            if node.points is not None:
                for pt in node.points:
                    if np.sqrt(self.dist_function(pt, center_point)) <= radius:
                        result.append(pt)
            else:
                _query(node.left)
                _query(node.right)
        _query(self.root)
        return result