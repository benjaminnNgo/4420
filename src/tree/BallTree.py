import os
import sys
import numpy as np
from typing import Optional
from typing import *
import heapq

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure


"""
Authors: Jared Rost
Overview: Implementation of Ball*-tree data structure.

This implementation follows the Ball*-tree algorithm described in:
"Metric Ball-tree: An Efficient Data Structure for Distance Queries in Metric Spaces"
http://arxiv.org/pdf/1511.00628

The Ball*-tree is a space-partitioning data structure that organizes points in a 
metric space using nested hyperspheres. It is especially efficient for nearest neighbor 
queries in high-dimensional spaces where traditional structures like KD-trees become 
less effective due to the "curse of dimensionality".

Key features of this implementation:
- PCA-based splitting for adaptive partitioning
- Lazy deletion for improved deletion performance
- Special handling for degenerate data distributions
- Optimized distance-based pruning for queries
"""

class BallTreeNode:
    r"""Node for the Ball*-tree.

    Each node represents a hypersphere containing a subset of points, with:
    - A center point defining the sphere's location
    - A radius encompassing all contained points
    - Left and right child nodes for internal nodes
    - A direct list of points for leaf nodes

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
        """
        Initialize a Ball*-tree node.
        
        Args:
            center (np.array): Center point of the bounding hypersphere.
            radius (float): Radius of the hypersphere, covering all contained points.
            left (Optional[BallTreeNode]): Left child node, contains points with 
                                          projections <= median on principal axis.
            right (Optional[BallTreeNode]): Right child node, contains points with 
                                           projections > median on principal axis.
            points (Optional[List[np.ndarray]]): Actual points stored (for leaf nodes only).
        """
        self.center = center
        self.radius = radius
        self.left = left
        self.right = right
        self.points = points

    def __sizeof__(self):
        """
        Calculate the memory usage of this node and its subtree in bytes.
        
        Accounts for:
        - The node's attributes
        - The center point and radius
        - All points in leaf nodes
        - All child nodes recursively
        
        Returns:
            int: Memory usage in bytes
        """
        size = sys.getsizeof(self.__dict__)
        size += sys.getsizeof(self.center)
        size += sys.getsizeof(self.radius)
        if self.left is not None:
            size += self.left.__sizeof__()
        if self.right is not None:
            size += self.right.__sizeof__()
        if self.points is not None:
            size += sys.getsizeof(self.points)
            for pt in self.points:
                size += sys.getsizeof(pt)
                size += pt.nbytes  # Account for the actual numpy array data
        return size

class BallTree(GeometricDataStructure):
    r"""Ball*-tree implementation for efficient nearest neighbor queries.
    
    This implementation uses hyperspheres to partition the space adaptively based on
    the data distribution. It performs Principal Component Analysis (PCA) at each split
    to find the direction of maximum variance, then divides points based on their
    projections onto this principal axis.
    
    Key features:
    - PCA-based splitting for efficient partitioning
    - Hierarchical hypersphere structure for space representation
    - Distance-based pruning for query acceleration
    - Lazy deletion for improved deletion performance
    - Special handling for degenerate data distributions
    
    Time complexity:
    - Construction: O(n log n) average case
    - Search: O(log n) average case, O(n) worst case
    - Insert/Delete: O(log n) average case, O(n) worst case for restructuring
    
    Args:
        dimension (int): Dimension of each point
        points (List[np.ndarray]): List of initial points
        dist_function (callable): Function to compute distance between 2 points,
                                 defaults to euclidean_squared_distance
    """
    def __init__(self, dimension, points=None, dist_function=None):
        """
        Initialize the Ball*-tree with the given points.
        
        Sets up a leaf size of 50 points, constructs the initial tree,
        and initializes lazy deletion tracking structures.
        """
        super().__init__(dimension, points, dist_function)
        self.leaf_size = 50  # Maximum points per leaf node
        self.root = self._construct_tree(points, depth=0)
        # Add tracking for deleted points
        self._deleted_points = set()  # Store tuples of deleted points
        self._deleted_count = 0
        self._rebuild_threshold = 0.25  # Rebuild when 25% of points are deleted
    
    def _point_to_tuple(self, point):
        """
        Convert numpy array to hashable tuple for tracking.
        
        Needed for storing points in sets for efficient lookup in lazy deletion.
        
        Args:
            point (np.ndarray): Point to convert
            
        Returns:
            tuple: Hashable tuple representation of the point
        """
        return tuple(point.tolist())
    
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
        # Base case: no points to split.
        if points is None or len(points) == 0:
            return None
        # Base case: leaf node.
        if len(points) <= self.leaf_size:
            center = np.mean(points, axis=0)
            radius = np.sqrt(max(self.dist_function(center, pt) for pt in points))
            return BallTreeNode(center=center, radius=radius, points=points)
        
        # Compute the center.
        center = np.mean(points, axis=0)
        pts_array = np.vstack(points)

        # Compute covariance and its principal eigenvector.
        cov_matrix = np.cov(pts_array.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        # Degenerate case: minimal variance in all dimensions.
        if np.max(eigenvalues) < 1e-6:
            radius = np.sqrt(max(self.dist_function(center, pt) for pt in points))
            return BallTreeNode(center=center, radius=radius, points=points)
        
        principal_idx = np.argmax(eigenvalues)
        principal_axis = eigenvectors[:, principal_idx]
        
        # Project points onto the principal axis and split at median.
        projections = [np.dot(pt - center, principal_axis) for pt in points]
        median_proj = np.median(projections)
        left_points = [pt for pt, proj in zip(points, projections) if proj <= median_proj]
        right_points = [pt for pt, proj in zip(points, projections) if proj > median_proj]
        
        # Handle failed splits by creating a leaf node.
        if len(left_points) == 0 or len(right_points) == 0 or \
           len(left_points) == len(points) or len(right_points) == len(points):
            radius = np.sqrt(max(self.dist_function(center, pt) for pt in points))
            return BallTreeNode(center=center, radius=radius, points=points)
        
        # Create internal node with two children.
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
        
        If a leaf node is reached:
        - If the leaf has space, add the point and update the ball
        - If the leaf is full, rebuild it as a subtree
        
        For internal nodes:
        - Update the bounding sphere if needed
        - Propagate the insertion to the closest child
        
        Args:
            node (BallTreeNode): Current node to insert into
            point (np.ndarray): Point to insert
            depth (int): Current recursion depth
            
        Returns:
            BallTreeNode: Updated node (may be rebuilt)
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
        Inserts a new point into the Ball*-tree.
        
        Time complexity: O(log n) average case, O(n) worst case when a subtree must be rebuilt
        
        Args:
            point (np.ndarray): Point to insert
        """
        self.root = self._insert(self.root, point, depth=0)
    
    def _collect_points(self, node: BallTreeNode) -> List[np.ndarray]:
        """
        Recursively collects all points from the subtree rooted at `node`.
        
        Args:
            node (BallTreeNode): Root of subtree to collect from
            
        Returns:
            List[np.ndarray]: All points in the subtree
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
        This is an expensive operation, which is why lazy_delete is preferred.
        
        Args:
            node (BallTreeNode): Current node to check
            point (np.ndarray): Point to delete
            depth (int): Current recursion depth
            
        Returns:
            Tuple[Optional[BallTreeNode], bool]: (Updated node, whether deletion occurred)
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
        
        Note: This is an expensive operation. Consider using lazy_delete instead.
        
        Time complexity: O(n) worst case due to subtree rebuilding
        
        Args:
            point (np.ndarray): Point to delete
            
        Raises:
            Exception: If point not found in the tree
        """
        self.root, deleted = self._delete(self.root, point, depth=0)
        if not deleted:
            raise Exception("Point not found in the Ball*-tree")
        
    def lazy_delete(self, point: np.ndarray):
        """
        Mark a point as deleted without restructuring the tree.
        
        This method avoids the expensive tree restructuring that occurs with 
        standard deletion by simply marking points as deleted in a separate set.
        The tree is only rebuilt when the number of deleted points exceeds a
        threshold proportion of the total points.
        
        Args:
            point (np.ndarray): The point to delete
            
        Returns:
            bool: True if point found and marked as deleted
            
        Raises:
            Exception: If point not found in the tree
        """
        point_tuple = self._point_to_tuple(point)
        
        # Skip if already deleted
        if point_tuple in self._deleted_points:
            return True
            
        # Check if point exists before marking deleted
        found = self._check_point_exists(self.root, point)
        if not found:
            raise Exception("Point not found in the Ball*-tree")
            
        # Mark as deleted
        self._deleted_points.add(point_tuple)
        self._deleted_count += 1
        
        # Rebuild if too many deleted points
        total_points = len(self._collect_all_points(self.root))
        if self._deleted_count > total_points * self._rebuild_threshold:
            self._cleanup()
            
        return True
    
    def _check_point_exists(self, node, point):
        """
        Check if a point exists in the tree without modifying structure.
        
        Uses sphere geometry to prune branches that cannot contain the point.
        
        Args:
            node (BallTreeNode): Current node to check
            point (np.ndarray): Point to search for
            
        Returns:
            bool: True if point exists, False otherwise
        """
        if node is None:
            return False
            
        if node.points is not None:
            return any(np.array_equal(pt, point) for pt in node.points)
            
        # Compute distances to children's centers
        left_dist = np.inf if node.left is None else np.sqrt(self.dist_function(node.left.center, point))
        right_dist = np.inf if node.right is None else np.sqrt(self.dist_function(node.right.center, point))
        
        # Pruning: only check a child if point could be within its radius
        if node.left is not None and left_dist <= node.left.radius:
            if self._check_point_exists(node.left, point):
                return True
                
        if node.right is not None and right_dist <= node.right.radius:
            if self._check_point_exists(node.right, point):
                return True
                
        return False
    
    def _collect_all_points(self, node):
        """
        Collect all non-deleted points from the subtree.
        
        Args:
            node (BallTreeNode): Root of subtree to collect points from
            
        Returns:
            List[np.ndarray]: List of all valid points in the subtree
        """
        if node is None:
            return []
            
        points = []
        if node.points is not None:
            for pt in node.points:
                if self._point_to_tuple(pt) not in self._deleted_points:
                    points.append(pt)
        else:
            points.extend(self._collect_all_points(node.left))
            points.extend(self._collect_all_points(node.right))
        return points
    
    def _cleanup(self):
        """
        Rebuild the tree without deleted points.
        
        Collects all non-deleted points and reconstructs the entire tree,
        then resets the deleted points tracking.
        """
        all_points = self._collect_all_points(self.root)
        self.root = self._construct_tree(all_points, 0)
        self._deleted_points.clear()
        self._deleted_count = 0

    def get_knn(self, point, k):
        """
        Get k nearest neighbors to the target point.
        
        Uses a distance-based pruning algorithm with priority queue to efficiently
        find the k nearest neighbors without examining all points.
        
        Args:
            point (np.ndarray): The query point
            k (int): Number of neighbors to retrieve
            
        Returns:
            List[np.ndarray]: k nearest points sorted by distance (closest first)
        """
        priority_queue, _ = self._get_knn(point, self.root, k, [])
        knn = [(-d, pt) for d, _, pt in priority_queue]
        knn.sort(key=lambda x: x[0])
        return [pt for _, pt in knn[:k]]
    
    def _get_knn(self, 
                 point: np.ndarray, 
                 node: BallTreeNode, 
                 k: int, 
                 priority_queue: List, 
                 depth: int = 0, 
                 tiebreaker: int = 1) -> Tuple[List, int]:
        r"""
        Recursively retrieves k nearest neighbors using a max-heap priority queue.
        
        Uses distance-based pruning to avoid exploring branches that cannot contain
        points closer than the current k-th nearest neighbor.
        
        Args:
            point (np.ndarray): Query point
            node (BallTreeNode): Current node being explored
            k (int): Number of neighbors to find
            priority_queue (List): Max heap of current best candidates
            depth (int): Current recursion depth
            tiebreaker (int): Value to break ties for equal distances
            
        Returns:
            Tuple[List, int]: (Updated priority queue, updated tiebreaker)
        """
        if node is None:
            return priority_queue, tiebreaker
        
        # Compute the lower bound based on the node's sphere.
        center_distance = np.sqrt(self.dist_function(node.center, point))
        lower_bound = max(0, center_distance - node.radius)
        # Prune if the lower bound exceeds the worst current candidate.
        if len(priority_queue) == k and lower_bound >= -priority_queue[0][0]:
            return priority_queue, tiebreaker
        
        # Leaf node: compare all points.
        if node.points is not None:
            for pt in node.points:
                # Skip deleted points
                if self._point_to_tuple(pt) in self._deleted_points:
                    continue
                    
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
        
        # Internal node: choose the child with the closer sphere.
        children = []
        if node.left is not None:
            left_center_distance = np.sqrt(self.dist_function(node.left.center, point))
            left_lb = max(0, left_center_distance - node.left.radius)
            children.append((left_lb, node.left))
        if node.right is not None:
            right_center_distance = np.sqrt(self.dist_function(node.right.center, point))
            right_lb = max(0, right_center_distance - node.right.radius)
            children.append((right_lb, node.right))
        children.sort(key=lambda x: x[0])  # Visit the closest lower bound first
        for lb, child in children:
            priority_queue, tiebreaker = self._get_knn(point, child, k, priority_queue, depth+1, tiebreaker)
        return priority_queue, tiebreaker

    def get_nearest(self, point: np.ndarray):
        r"""
        Retrieves the nearest neighbor of a target point.
        
        Time complexity: O(log n) average case
        
        Args:
            point (np.ndarray): Query point
            
        Returns:
            np.ndarray: The nearest point, or None if the tree is empty
        """
        knn = self.get_knn(point, k=1)
        return knn[0] if knn else None

    def query_range(self,
                center_point: np.ndarray, 
                radius: float):
        r"""
        Retrieves all points within a given radius of center_point.
        
        Uses distance-based pruning to efficiently search only relevant branches.
        
        Args:
            center_point (np.ndarray): The center of the search sphere
            radius (float): The search radius
            
        Returns:
            List[np.ndarray]: Points within the radius
        """
        result = []
        # Square the radius once for comparison with squared distances
        radius_squared = radius**2
        
        def _query(node: BallTreeNode):
            """Inner recursive function for range queries"""
            if node is None:
                return
                
            # Use squared distance directly without square root
            center_dist_squared = self.dist_function(node.center, center_point)
            
            # Pruning condition using squared distance
            # If the minimum possible squared distance exceeds radius_squared, prune
            # Calculate the minimum possible distance from query point to any point in this node
            min_possible_dist = max(0, np.sqrt(center_dist_squared) - node.radius)
            if min_possible_dist**2 > radius_squared:
                return
                
            if node.points is not None:
                for pt in node.points:
                    # Skip deleted points
                    if self._point_to_tuple(pt) in self._deleted_points:
                        continue
                        
                    # Direct comparison of squared distances
                    if self.dist_function(pt, center_point) <= radius_squared:
                        result.append(pt)
            else:
                _query(node.left)
                _query(node.right)
                
        _query(self.root)
        return result
        
    def __sizeof__(self):
        """
        Calculate the memory usage of the Ball*-tree in bytes.
        
        Accounts for:
        - The Ball*-tree object attributes
        - The root node and all its children recursively
        - The deleted points tracking structures
        
        Returns:
            int: Memory usage in bytes
        """
        size = sys.getsizeof(self.__dict__)
        size += sys.getsizeof(self.dimension)
        size += sys.getsizeof(self.leaf_size)
        size += sys.getsizeof(self._deleted_points)
        size += sys.getsizeof(self._deleted_count)
        size += sys.getsizeof(self._rebuild_threshold)
        
        if self.root is not None:
            size += self.root.__sizeof__()
            
        for point_tuple in self._deleted_points:
            size += sys.getsizeof(point_tuple)
            
        return size
