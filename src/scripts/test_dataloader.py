import h5py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from data_loader import data_loader
from tree import KDTree, BruteForce
from tree.utils import euclidean_distance

# # Open the HDF5 file in read mode
# with h5py.File("datasets/notre.hdf5", "r") as f:
#     # List all groups and datasets in the file
#     print(list(f.keys()))
    
#     # Access a dataset
#     dataset = f["dataset"][0]
#     print(len(dataset))

data = data_loader()
print(data.dim)
print(len(data.points[0]))

test_tree = KDTree(dimension=data.dim, points=data.points)

point = test_tree.get_nearest(data.queries[0])
