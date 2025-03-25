import os
import random
import sys
import numpy as np
from typing import Optional
from typing import *
import heapq
from collections import deque

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tree import GeometricDataStructure

"""
Authors: Tran Gia Bao Ngo
Overview: Implementation of KD-tree from the `"Multidimensional binary search trees used for associative searching" <https://doi.org/10.1145/361002.361007>
"""

class KDTreeNode:
    r"""Representing a node in KD-tree

    Args:
        coordinate (np.array): 1D vector recording coordinates of a point
        left (KDTreeNode) : left child
        right (KDTreeNode) : right child
    """

    def __init__(self,
                 coordinate:np.array,
                 left: Optional["KDTreeNode"] = None,
                 right: Optional["KDTreeNode"] = None
                ):
        
        self.coordinate = coordinate
        self.right = right
        self.left = left

    def __sizeof__(self):
        size = sys.getsizeof(self.__dict__)
        size += sys.getsizeof(self.coordinate)
        if hasattr(self.coordinate, "nbytes"):
            size += self.coordinate.nbytes
        if self.left is not None:
            size += self.left.__sizeof__()  # Call __sizeof__ recursively, not sys.getsizeof()
        if self.right is not None:
            size += self.right.__sizeof__()  # Call __sizeof__ recursively, not sys.getsizeof()
        return size
        
class KDTree(GeometricDataStructure):
    r"""KD-tree from the `"Multidimensional binary search trees used for associative searching" <https://doi.org/10.1145/361002.361007>`

    Args:
        dimension (int): dimension of each point
        points (List[np.ndarray]) : list of initial points
        dist_function (function pointer) : function to compute distance between 2 points, by `default` is `euclidean_squ_distance`
    """
    def __init__(self,
                 dimension : int,
                 points:Optional[List[np.ndarray]] = None,
                 dist_function : Optional[Callable] = None 
                ):
        super().__init__(dimension,points,dist_function)
        self.root = self._construct_tree(points=points, depth=0)

    def __sizeof__(self):
        size = sys.getsizeof(self.__dict__)
        size += sys.getsizeof(self.dimension)
        if self.root is not None:
            size += self.root.__sizeof__()  # Call __sizeof__ recursively, not sys.getsizeof()
        return size

    def _construct_tree(self,
                        points: List[np.ndarray], 
                        depth = 0):
        r"""
        Private function: constructing an tree from scratch given list of points recursively

        Args:
            points (List[np.ndarray]) : list of points
            depth (int) : indicate current depth level when call this function. This is used to define which axis to compare
        """
        if not points:
            return None
        
        compared_axis = depth % self.dimension #define which axis used to split tree into left and right branches

        median_point = quickselect_median_point(points,compared_axis) #Get the median node to split list of nodes into 2 approximately equal 2 sublist
        left = [point for point in points if point[compared_axis] <= median_point[compared_axis] and np.any(point != median_point)]
        right = [point for point in points if point[compared_axis] > median_point[compared_axis]]
        new_node = KDTreeNode(coordinate= median_point)
        new_node.left = self._construct_tree(points= left, depth= depth+1)
        new_node.right = self._construct_tree(points= right, depth= depth+1)
        return new_node

        
    def _insert(self, 
                root:KDTreeNode, 
                point:np.ndarray,
                depth:int = 0):
        r"""
        Private function: Given current node in a tree and a new point, decide whether to insert into left branch or right branch (recursively)
        If left/right child of current node is None (base case), insert directly 

        Args:
            root (KDTreeNode) : current node in the tree
            point (np.ndarray) : inserting point
            depth (int) : indicate current depth level when call this function. This is used to define which axis to compare
        """
        compared_axis = depth % self.dimension
        dx =  point[compared_axis] - root.coordinate[compared_axis]
        if dx > 0:
            if root.right is None:
                root.right =  KDTreeNode(coordinate= point)
            else:
                self._insert(root= root.right,point=point,depth=depth+1)
        else:

            if root.left is None:
                root.left =  KDTreeNode(coordinate= point)
            else:
                self._insert(root= root.left,point=point,depth=depth+1)

    def _find_min(self,
                  node: KDTreeNode,
                  target_dim: int, 
                  depth:int = 0):
        r"""
        Find j-minimum in `node`'s subtree
        """
        curr_node_discriminator  = depth % self.dimension
        if node is None:
            return None

        if target_dim == curr_node_discriminator:
            if node.left is None:
                return node
            return self._find_min(node = node.left, target_dim= target_dim, depth= depth + 1 )
        else:
            left_min = self._find_min(node.left, target_dim=target_dim, depth=depth + 1)
            right_min = self._find_min(node.right, target_dim=target_dim, depth=depth + 1)
            not_none_candidates = [node for node in [node, left_min, right_min] if node is not None] #Filter out left_min and right_min if it is none
            return min(not_none_candidates, key= lambda node: node.coordinate[target_dim])
        
    
    def _find_max(self,
                  node: KDTreeNode,
                  target_dim: int, 
                  depth:int = 0):
        r"""
        Find j-maximum in `node`'s subtree
        """
        curr_node_discriminator  = depth % self.dimension
        if node is None:
            return None

        if target_dim == curr_node_discriminator:
            if node.right is None:
                return node
            return self._find_max(node = node.right, target_dim= target_dim, depth= depth + 1 )
        else:
            left_max = self._find_max(node.left, target_dim=target_dim, depth=depth + 1)
            right_max = self._find_max(node.right, target_dim=target_dim, depth=depth + 1)
            not_none_candidates = [node for node in [node, left_max, right_max] if node is not None] #Filter out left_min and right_min if it is none
            return max(not_none_candidates, key= lambda node: node.coordinate[target_dim])
        
    def _delete(self,
                root:KDTreeNode, 
                target_point: np.ndarray,
                depth: int = 0):
        if root is None:
            return None
        r"""
        Delete a point from a subtree starting with given root (This function will be call recursively)
        
        Args:
            - root (KDTreeNode) : the starting node of the subtree
            - target_point (np.ndarray) : point we want to delete
            - depth (int) : depth of current root, which is used to determine discriminator
        """        
        curr_node_discriminator = depth %self.dimension
        if np.array_equal(target_point, root.coordinate): 
            # Found the node to delete
            if root.right is None and root.left is None:
                return None
            
            if root.right is not None:
                min_node = self._find_min(node= root.right,target_dim=curr_node_discriminator, depth= depth +1)
                root.coordinate = min_node.coordinate # replace root with j-min from right subtree
                root.right = self._delete(root= root.right, target_point= min_node.coordinate,depth=depth + 1)  # deleting min_node from right subtree
            else:
                max_node = self._find_max(node= root.left,target_dim=curr_node_discriminator, depth= depth +1)
                root.coordinate = max_node.coordinate # replace root with j-max from left subtree
                root.left = self._delete(root= root.left, target_point= max_node.coordinate,depth=depth + 1) # deleting max_node from left subtree
                
            
        else:
            # Not able to find node to delete yet, keep traverse the tree
            if target_point[curr_node_discriminator] >= root.coordinate[curr_node_discriminator]:
                root.right = self._delete(root= root.right,target_point= target_point, depth= depth + 1)
            else:
                root.left = self._delete(root= root.left,target_point= target_point, depth= depth + 1)
        
        return root


    def insert(self,
               point:np.ndarray): 
        r"""
        Insert a new node to the tree

        Args:
            point (np.ndarray) : new node
        """
        
        if self.root is None:
            self.root = KDTreeNode(coordinate= point)
        else:
            self._insert(root= self.root, point = point,depth = 0)

    def delete(self,
               point : np.ndarray): 
        r"""
        Delete a given node from the data structure

        Args:
            point (np.ndarray) : target point
        """
        self._delete(root=self.root,target_point=point,depth= 0)



    def _get_knn(self, 
                 point:np.ndarray, 
                 compared_node: KDTreeNode, 
                 k: int, 
                 priority_queue: List, 
                 depth: int = 0, 
                 tiebreaker = 1):
        r"""
        Private function: Get k nearest neighbours recursively 

        Args:
            point (np.ndarray) : target point 
            compared_node (KDTreeNode) : current node in the tree
            k (int) : indicate the number of neighbours to query 
            priority_queue (List): priority queue to keep track of k nearest neighbours 
            depth (int) : depth of current node in the tree. This is used to define which axis to compare 
            tiebreaker (int): When 2 points having the same distance to target points, `heapq` will use this value to
                              define priority. In addition,`tiebreaker` for each element is unique. All of these to 
                              avoid randomness made by `heapq` in return results.
        """
        
        if compared_node is None: # Reach the end of the tree
            return
        
        compared_axis = depth % self.dimension
        curr_dist = self.dist_function(pointA= point, pointB= compared_node.coordinate)

        if len(priority_queue) < k:
            heapq.heappush(priority_queue,(-curr_dist,tiebreaker,compared_node))
        elif curr_dist < -priority_queue[0][0]:
            heapq.heappushpop(priority_queue,(-curr_dist,tiebreaker,compared_node)) 
            """
            The above line of code is equivalent to:
            heapq.heappop(priority_queue)
            heapq.heappush(priority_queue,(-square_dist,tiebreaker,compared_node)) 

            but for efficiency, we call heappushpop, which accomplish the same thing but faster
            """
        

        if point[compared_axis] < compared_node.coordinate[compared_axis]:
            # We prioritize go the left child in this case
            next_node = compared_node.left
            other_node = compared_node.right
        else:
            # We prioritize go the right child in this case
            next_node = compared_node.right
            other_node = compared_node.left
        
        self._get_knn(point=point, compared_node= next_node, k= k, priority_queue = priority_queue, depth = depth + 1, tiebreaker= 2*tiebreaker)
        if (point[compared_axis] - compared_node.coordinate[compared_axis])**2 < -priority_queue[0][0]:
            # We need to traverse other branch since we can find a closer node potentially in this case
            self._get_knn(point=point, compared_node= other_node, k= k, priority_queue = priority_queue, depth = depth + 1, tiebreaker= 2*tiebreaker + 1)

        if tiebreaker == 1:
            return [queue_element[2].coordinate.tolist() for queue_element in sorted(priority_queue, reverse= True)]
    
    def get_knn(self,
                point: np.ndarray,
                k:int): 
        r"""
        Get k nearest neighbours 

        Args:
            point (np.ndarray) : target point 
            k (int) : indicate the number of neighbours to query 
        """
        return self._get_knn(point = point, compared_node = self.root, k= k, priority_queue = [])    
    
    def get_nearest(self,
                    point : np.ndarray): 
        r"""
        Get the nearest neighbours 

        Args:
            point (np.ndarray) : target point 
        """
        return self._get_knn(point = np.array(point), compared_node = self.root, k= 1, priority_queue = [])[0]
    
    def _range_search(self, 
                 point: np.ndarray, 
                 radius: float,
                 node: KDTreeNode, 
                 results: List,
                 depth: int = 0):
        """
        Private function: Find all points within the specified radius of the target point
        
        Args:
            point (np.ndarray): target point
            radius (float): search radius
            node (KDTreeNode): current node in the tree
            results (List): accumulator for points within the radius
            depth (int): depth of current node, used to determine the axis
        """
        if node is None:
            return
        
        # Calculate the distance from the query point to the current node's point
        curr_dist = self.dist_function(pointA=point, pointB=node.coordinate)
        
        # If this point is within the radius, add it to our results
        if curr_dist <= radius**2:  # Compare with squared radius since we're using squared distance
            results.append(node.coordinate.tolist())
        
        # Determine the axis for this level
        curr_axis = depth % self.dimension
        
        # Determine which child to visit first (the one on the same side of the splitting plane)
        if point[curr_axis] < node.coordinate[curr_axis]:
            first_branch = node.left
            second_branch = node.right
        else:
            first_branch = node.right
            second_branch = node.left
        
        # Always search the branch that contains the query point
        self._range_search(point, radius, first_branch, results, depth + 1)
        
        # Check if we need to search the other branch
        # We only need to search it if the distance from the query point to the splitting plane
        # is less than or equal to the radius
        if (point[curr_axis] - node.coordinate[curr_axis])**2 <= radius**2:
            self._range_search(point, radius, second_branch, results, depth + 1)
        
        return results

    def query_range(self, center_point: np.ndarray, radius: float):
        """
        Find all points within the specified radius of the target point
        
        Args:
            point (np.ndarray): target point
            radius (float): search radius
            
        Returns:
            List: all points within the radius
        """
        return self._range_search(center_point, radius, self.root, [], 0)
    
    def print_tree(self):
        if self.root is None:
            return
        
        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            for _ in range(level_size):
                curr_node = queue.popleft()
                print(curr_node.coordinate,end="")
                child_indicator = 0
                if curr_node.left is not None:
                    queue.append(curr_node.left)
                    child_indicator +=1
                
                if curr_node.right is not None:
                    queue.append(curr_node.right)
                    child_indicator +=2
                print(f"({child_indicator})",end="\t")
                
            print()

def quickselect_median_point(points:List, 
                             dim :int = 0,
                             select_pivot_fn: Optional[Callable]= random.choice):
    r"""
    Finding median point from list of high dimensional points with average time complexity O(n)
    Implemented based on quick select algorithm

    Args:
        points (list) : list of high dimensional point
        dim (int) : determine which dimension to compare
        select_pivot_fn (function pointer) : determine how to select pivot
    """
    return quickselect(points, len(points) // 2, select_pivot_fn,dim)
    

def quickselect(points:list, 
                k:int, 
                select_pivot_fn: Callable,
                dim:int):
    
    r"""
    Quick select algorithm

    Args:
        points (list) : list of high dimensional point
        k (int) : kth largest largest element
        select_pivot_fn (function pointer) : determine how to select pivot
        dim (int) : determine which dimension to compare
    """

    if len(points) == 1:
        assert k == 0
        return points[0]

    pivot = select_pivot_fn(points)

    lows = [el for el in points if el[dim] < pivot[dim]]
    highs = [el for el in points if el[dim] > pivot[dim]]
    pivots = [el for el in points if el[dim] == pivot[dim]]

    if k < len(lows):
        return quickselect(lows, k, select_pivot_fn,dim)
    elif k < len(lows) + len(pivots):
        # Find kth largest element
        return pivots[0]
    else:
        return quickselect(highs, k - len(lows) - len(pivots), select_pivot_fn,dim)