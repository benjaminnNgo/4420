from typing import *
import os
import sys
import h5py
import numpy as np

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants
MAX_DATASET_SIZE = 50000
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
DATASETS_DIR = os.path.join("src", "datasets") 

class GeometricData:
    def __init__(self,
                 dim: int,
                 points: List[np.ndarray],
                 queries: List[List],
                 groundtruth_queries: List[List],
                 random_queries: List[List],
                 groundtruth_random_queries : List[List]):
        
        self.dim = dim
        self.points = points
        self.queries = queries
        self.groundtruth_queries = groundtruth_queries
        self.random_queries = random_queries
        self.groundtruth_random_queries = groundtruth_random_queries
    
def data_loader(data_name = "cifar", cap_dataset = False):
    r"""
    Datasets are available to download from
    https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data

    Please make sure to download datasets into `datasets` dir before this function to process the data
    Available datasets:
        - cifar
        - audio
        - deep
        - enron
        - glove
        - imageNet
        - millionSong
        - MNIST
        - notre
        - nuswide
        - sift
        - sun
        - trevi
        - ukbench
        
    Note: 
        - Queries are limited to 200 points maximum.
        - When cap_dataset=True, dataset is limited to 50,000 points maximum.
    """
    
    if data_name not in AVAILABLE_DATASETS:
        raise Exception("Unsupport dataset {}".format(data_name))
    
    data_path = f"{DATASETS_DIR}/{data_name}.hdf5"
    if not os.path.exists(data_path):
        raise Exception(f"Please download {data_name}.hdf5 from https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data into datasets directory")
    
    data = None
    # Open the HDF5 file in read mode
    with h5py.File(data_path, "r") as datafile:
        # Load dataset with optional capping
        dataset = datafile["dataset"]
        if cap_dataset:
            dataset_size = min(MAX_DATASET_SIZE, dataset.shape[0])
            dataset = list(dataset[:dataset_size])
        else:
            dataset = list(dataset[:])

        # Cap queries to max 200 points
        queries = datafile['query']
        query_size = min(200, queries.shape[0])
        queries = list(queries[:query_size])

        # Ensure groundtruth is consistent with number of queries
        groundtruth_queries = datafile['groundtruth']
        groundtruth_queries = list(groundtruth_queries[:query_size])
        
        # Also cap random queries and their groundtruth
        random_queries = datafile['randomquery']
        random_queries = list(random_queries[:query_size])
        
        groundtruth_random_queries = datafile['groundtruth_randomquery']
        groundtruth_random_queries = list(groundtruth_random_queries[:query_size])

        data = GeometricData(
            dim=dataset[0].shape[0],
            points=dataset,
            queries=queries,
            groundtruth_queries=groundtruth_queries,
            random_queries=random_queries,
            groundtruth_random_queries=groundtruth_random_queries
        )
    return data
