from typing import *
import os
import sys
import h5py
import numpy as np

# Edit path to import from different module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


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
    
def data_loader(data_name = "cifar"):
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
}
    """
    
    if data_name not in AVAILABLE_DATASETS:
        raise Exception("Unsupport dataset {}".format(data_name))
    
    data_path = f"{DATASETS_DIR}/{data_name}.hdf5"
    if not os.path.exists(data_path):
        raise Exception(f"Please download {data_name}.hdf5 from https://github.com/DBAIWangGroup/nns_benchmark/tree/master/data into datasets directory")
    
    data = None
    # Open the HDF5 file in read mode
    with h5py.File(data_path, "r") as datafile:
        dataset = datafile["dataset"]
        queries = datafile['query']
        groundtruth_queries = datafile['groundtruth']
        random_queries = datafile['randomquery']
        groundtruth_random_queries = datafile['groundtruth_randomquery']

        dataset = list(dataset[:])   

        data = GeometricData(
            dim=dataset[0].shape[0],
            points= list(dataset[:]),
            queries = list(queries[:]),
            groundtruth_queries = list(groundtruth_queries[:]),
            random_queries = list(random_queries[:]),
            groundtruth_random_queries = list(groundtruth_random_queries[:])
        )
    return data
    
    
