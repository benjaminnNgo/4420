"""
Gaussian Mixture Dataset Generator

This script generates a synthetic dataset composed of Gaussian mixture clusters in high-dimensional space.
The dataset is particularly useful for benchmarking different nearest neighbor search algorithms,
especially in high dimensions where the curse of dimensionality affects performance. We generated our own Gauss
dataset since the one from the original paper was not available.

Key characteristics of the generated dataset:
- Points are grouped into clusters with Gaussian distribution around center points
- Most variance is in the first 5 dimensions, with small noise in the remaining dimensions
- This design creates a dataset that mimics real-world high-dimensional data where
  only a subset of dimensions contain most of the meaningful information

The output is saved as an HDF5 file with the following structure:
- dataset: Main point dataset
- query: Points to use for querying the data structures
- groundtruth: (Dummy data, not actually used)
- randomquery: (Dummy data, not actually used)
- groundtruth_randomquery: (Dummy data, not actually used)
"""

import numpy as np
import h5py
import os
import sys

def generate_gauss_dataset(n=10, seed=42, output_dir=None):
    """
    Generate a Gaussian mixture dataset for benchmarking nearest neighbor algorithms.
    
    The dataset consists of points from Gaussian clusters where the first 5 dimensions
    contain most of the signal, and the remaining dimensions have small random noise.
    This mimics real-world high-dimensional data where only a few dimensions contain
    most of the relevant information.
    
    Args:
        n (int): Scaling factor for the number of points (2000 * n^3 total points).
                 Higher values create larger datasets.
        seed (int): Random seed for reproducibility of the generated data.
        output_dir (str): Directory where to save the HDF5 file. If None, defaults
                          to the "../datasets" relative to the script location.
    """
    # Resolve the output directory path if not explicitly provided
    if output_dir is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the src directory, then to datasets
        output_dir = os.path.join(os.path.dirname(script_dir), "datasets")
    np.random.seed(seed)
    
    # Dataset parameters
    num_clusters = 1000  # Number of Gaussian clusters
    num_points_total = 2000 * (n ** 3)  # Total dataset size scales with n^3
    points_per_cluster = num_points_total // num_clusters  # Evenly distribute points
    dim = 512  # Total dimensionality of the dataset
    num_queries = 200  # Number of query points to generate
    
    # Print dataset generation parameters for verification
    print(f"Generating Gaussian mixture dataset with:")
    print(f"- {num_points_total} total points")
    print(f"- {num_clusters} clusters")
    print(f"- {points_per_cluster} points per cluster")
    print(f"- {dim} dimensions")
    print(f"- {num_queries} query points")
    
    # Generate cluster centers in the first 5 dimensions
    # Centers are uniformly distributed in [0, 10]^5 space
    cluster_centers_5d = np.random.uniform(0, 10, size=(num_clusters, 5))
    
    # Allocate memory for the full dataset
    dataset = np.zeros((num_points_total, dim))
    current_idx = 0
    
    # Generate points for each cluster
    for i in range(num_clusters):
        # Progress indicator for large datasets
        if i % 100 == 0:
            print(f"Generating cluster {i}/{num_clusters}")
        
        # Get the center for this cluster
        center_5d = cluster_centers_5d[i]
        
        # Generate points around this center with standard deviation 1.0
        # These points follow a normal distribution in each of the 5 dimensions
        cluster_points_5d = np.random.normal(
            loc=center_5d,  # Center (mean) of the distribution
            scale=1.0,      # Standard deviation of the distribution
            size=(points_per_cluster, 5)  # Generate points_per_cluster samples in 5D
        )
        
        # Create full 512D points by extending the 5D points
        cluster_points = np.zeros((points_per_cluster, dim))
        
        # Copy the meaningful 5D data into the first 5 dimensions
        cluster_points[:, :5] = cluster_points_5d
        
        # Add small random noise to remaining dimensions
        # The small scale (0.1) ensures these dimensions have minimal impact compared to the first 5
        cluster_points[:, 5:] = np.random.normal(0, 0.1, size=(points_per_cluster, dim-5))
        
        # Add the generated points to the dataset
        dataset[current_idx:current_idx+points_per_cluster] = cluster_points
        current_idx += points_per_cluster
    
    # Generate query points that follow the same distribution as the dataset
    # These will be used to benchmark search performance
    query = np.zeros((num_queries, dim))
    
    # Sample from the same cluster centers but generate new points
    # This ensures query points are distributed similarly to the dataset
    query_cluster_indices = np.random.choice(num_clusters, num_queries, replace=True)
    
    print("Generating query points...")
    for i, cluster_idx in enumerate(query_cluster_indices):
        # Get the center of the randomly selected cluster
        center_5d = cluster_centers_5d[cluster_idx]
        
        # Sample a new point from this cluster's distribution
        query_point_5d = np.random.normal(loc=center_5d, scale=1.0, size=5)
        
        # Create the full 512D query point with the same structure as dataset points
        query[i, :5] = query_point_5d
        query[i, 5:] = np.random.normal(0, 0.1, size=dim-5)
    
    # Create placeholder data for compatibility with other benchmarks
    # These fields are not used in our benchmarks but included for format compatibility
    groundtruth = np.zeros((500,), dtype=np.int32)
    randomquery = np.zeros((500, dim))
    groundtruth_randomquery = np.zeros((500,), dtype=np.int32)
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gauss.hdf5")
    
    # Save everything to an HDF5 file with compression to reduce file size
    print(f"Saving dataset to {output_path}")
    with h5py.File(output_path, "w") as f:
        # Main dataset - use float32 precision to save space and enable compression
        f.create_dataset("dataset", data=dataset, dtype='float32', 
                        compression="gzip", compression_opts=4)
        
        # Query points - also compressed
        f.create_dataset("query", data=query, dtype='float32',
                        compression="gzip", compression_opts=4)
        
        # Dummy data - using appropriate dtypes
        f.create_dataset("groundtruth", data=groundtruth, dtype='int16')
        f.create_dataset("randomquery", data=randomquery, dtype='float32', 
                        compression="gzip", compression_opts=4)
        f.create_dataset("groundtruth_randomquery", data=groundtruth_randomquery, 
                        dtype='int16')
    
    # Report success and file size information
    print(f"Dataset successfully saved to {output_path}")
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # Default parameters
    n = 10  # Default scaling factor - n=10 creates a moderately sized dataset
    seed = 42  # Default random seed for reproducibility
    output_dir = None  # Use the default path resolution in the function
    
    # Parse command line arguments to allow customization
    if len(sys.argv) > 1:
        n = int(sys.argv[1])  # First arg: scaling factor
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])  # Second arg: random seed
    if len(sys.argv) > 3:
        output_dir = os.path.abspath(sys.argv[3])  # Third arg: output directory
    
    # Generate the dataset with the specified parameters
    generate_gauss_dataset(n, seed, output_dir)