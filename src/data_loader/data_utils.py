"""
Data Utilities for Geometric Data Structures Benchmarking

This module provides utilities for loading benchmark datasets stored in HDF5 files.
It defines:
  - Constants for available datasets and dataset size limits.
  - The GeometricData class which wraps loaded data (points, queries, and groundtruth).
  - The data_loader function that reads an HDF5 file and returns a GeometricData instance.

Datasets should be downloaded from:
    https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data
and placed in the defined DATASETS_DIR.
"""

from typing import *
import os
import sys
import h5py
import numpy as np

# Edit sys.path to allow relative importing from the parent directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants
MAX_DATASET_SIZE = 50000  # Maximum number of points to load when capping the dataset.
AVAILABLE_DATASETS = {
    'cifar',
    'audio',
    'deep',
    'enron',
    'glove',
    'imageNet',
    'millionSong',
    'gauss',
    'MNIST',
    'notre',
    'nuswide',
    'sift',
    'sun',
    'trevi',
    'ukbench'
}
DATASETS_DIR = os.path.join("src", "datasets")  # Directory where dataset HDF5 files are stored.

class GeometricData:
    """
    A wrapper class for geometric dataset information.

    Attributes:
        dim (int): The dimensionality of the data points.
        points (List[np.ndarray]): The list of data points.
        queries (List[List]): The list of query points.
        groundtruth_queries (List[List]): Groundtruth results for the query operations.
        random_queries (List[List]): List of randomly selected queries.
        groundtruth_random_queries (List[List]): Groundtruth for random queries.
    """
    def __init__(self,
                 dim: int,
                 points: List[np.ndarray],
                 queries: List[List],
                 groundtruth_queries: List[List],
                 random_queries: List[List],
                 groundtruth_random_queries: List[List]):
        self.dim = dim
        self.points = points
        self.queries = queries
        self.groundtruth_queries = groundtruth_queries
        self.random_queries = random_queries
        self.groundtruth_random_queries = groundtruth_random_queries

def data_loader(data_name: str = "cifar", cap_dataset: bool = False) -> GeometricData:
    r"""
    Loads a benchmark dataset from an HDF5 file.

    The HDF5 file is expected to contain at least the following datasets:
      - "dataset": The main data points.
      - "query": Query points for neighbor search.
      - "groundtruth": Groundtruth for queries.
      - "randomquery": Randomly chosen queries.
      - "groundtruth_randomquery": Groundtruth for random queries.

    Available datasets are the ones mentioned in AVAILABLE_DATASETS.
    Note:
      - Queries are limited to a maximum of 200 points.
      - When cap_dataset is True, the dataset is loaded with at most MAX_DATASET_SIZE points.

    Args:
        data_name (str): Name of the dataset to load (default: "cifar").
        cap_dataset (bool): Whether to limit the dataset to MAX_DATASET_SIZE (default: False).

    Returns:
        GeometricData: An instance containing the loaded dataset and associated data.

    Raises:
        Exception: If data_name is not in AVAILABLE_DATASETS or the HDF5 file is missing.
    """
    if data_name not in AVAILABLE_DATASETS:
        raise Exception("Unsupported dataset {}".format(data_name))
    
    data_path = f"{DATASETS_DIR}/{data_name}.hdf5"
    if not os.path.exists(data_path):
        raise Exception(f"Please download {data_name}.hdf5 from "
                        "https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data "
                        "into the datasets directory")
    
    data = None
    # Open the HDF5 file in read mode.
    with h5py.File(data_path, "r") as datafile:
        # Load the main dataset, applying capping if required.
        dataset = datafile["dataset"]
        if cap_dataset:
            dataset_size = min(MAX_DATASET_SIZE, dataset.shape[0])
            dataset = list(dataset[:dataset_size])
        else:
            dataset = list(dataset[:])
        
        # Load queries and cap to a maximum of 200.
        queries = datafile['query']
        query_size = min(200, queries.shape[0])
        queries = list(queries[:query_size])
        
        # Load groundtruth for queries.
        groundtruth_queries = datafile['groundtruth']
        groundtruth_queries = list(groundtruth_queries[:query_size])
        
        # Load random queries and their groundtruth.
        random_queries = datafile['randomquery']
        random_queries = list(random_queries[:query_size])
        
        groundtruth_random_queries = datafile['groundtruth_randomquery']
        groundtruth_random_queries = list(groundtruth_random_queries[:query_size])
        
        # Create an instance of GeometricData using the loaded information.
        data = GeometricData(
            dim=dataset[0].shape[0],
            points=dataset,
            queries=queries,
            groundtruth_queries=groundtruth_queries,
            random_queries=random_queries,
            groundtruth_random_queries=groundtruth_random_queries
        )
    return data
