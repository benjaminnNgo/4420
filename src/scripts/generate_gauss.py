import numpy as np
import h5py
import os
import sys

def generate_gauss_dataset(n=10, seed=42, output_dir=None):
    """
    Generate a Gaussian mixture dataset similar to the one described in the repository.
    
    Args:
        n (int): Scaling factor for the number of points (2000 * n^3 total points)
        seed (int): Random seed for reproducibility
        output_dir (str): Directory where to save the HDF5 file
    """
    # Use absolute path resolution
    if output_dir is None:
        # Get the directory of the current script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up one level to the src directory, then to datasets
        output_dir = os.path.join(os.path.dirname(script_dir), "datasets")
    np.random.seed(seed)
    
    # Dataset parameters
    num_clusters = 1000
    num_points_total = 2000 * (n ** 3)  # Total number of points
    points_per_cluster = num_points_total // num_clusters
    dim = 512  # Dimensionality of the data
    num_queries = 200  # Number of query points
    
    print(f"Generating Gaussian mixture dataset with:")
    print(f"- {num_points_total} total points")
    print(f"- {num_clusters} clusters")
    print(f"- {points_per_cluster} points per cluster")
    print(f"- {dim} dimensions")
    print(f"- {num_queries} query points")
    
    # Generate cluster centers in [0, 10]^5 space
    cluster_centers_5d = np.random.uniform(0, 10, size=(num_clusters, 5))
    
    # Generate points for each cluster
    dataset = np.zeros((num_points_total, dim))
    current_idx = 0
    
    for i in range(num_clusters):
        if i % 100 == 0:
            print(f"Generating cluster {i}/{num_clusters}")
        
        center_5d = cluster_centers_5d[i]
        # Generate 5D points around this center with std deviation 1.0
        cluster_points_5d = np.random.normal(
            loc=center_5d, 
            scale=1.0, 
            size=(points_per_cluster, 5)
        )
        
        # Create full 512D points
        cluster_points = np.zeros((points_per_cluster, dim))
        # Set the first 5 dimensions based on the Gaussian clusters
        cluster_points[:, :5] = cluster_points_5d
        
        # Add small random noise to remaining dimensions to make them meaningful
        cluster_points[:, 5:] = np.random.normal(0, 0.1, size=(points_per_cluster, dim-5))
        
        dataset[current_idx:current_idx+points_per_cluster] = cluster_points
        current_idx += points_per_cluster
    
    # Generate query points by sampling from the same clusters but creating new points
    query = np.zeros((num_queries, dim))
    
    # Choose random clusters to sample from (with replacement)
    query_cluster_indices = np.random.choice(num_clusters, num_queries, replace=True)
    
    print("Generating query points...")
    for i, cluster_idx in enumerate(query_cluster_indices):
        # Get the center of the selected cluster
        center_5d = cluster_centers_5d[cluster_idx]
        
        # Sample a new point from this cluster's Gaussian distribution
        query_point_5d = np.random.normal(loc=center_5d, scale=1.0, size=5)
        
        # Create full 512D query point
        query[i, :5] = query_point_5d
        query[i, 5:] = np.random.normal(0, 0.1, size=dim-5)
    
    # Create dummy data for other fields (not used)
    groundtruth = np.zeros((500,), dtype=np.int32)
    randomquery = np.zeros((500, dim))
    groundtruth_randomquery = np.zeros((500,), dtype=np.int32)
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "gauss.hdf5")
    
    # Create HDF5 file
    print(f"Saving dataset to {output_path}")
    # Save with float32 precision and compression
    with h5py.File(output_path, "w") as f:
        # Use float32 instead of float64, and enable compression
        f.create_dataset("dataset", data=dataset, dtype='float32', 
                        compression="gzip", compression_opts=4)
        f.create_dataset("query", data=query, dtype='float32',
                        compression="gzip", compression_opts=4)
        # Use int16 for groundtruth arrays
        f.create_dataset("groundtruth", data=groundtruth, dtype='int16')
        f.create_dataset("randomquery", data=randomquery, dtype='float32', 
                        compression="gzip", compression_opts=4)
        f.create_dataset("groundtruth_randomquery", data=groundtruth_randomquery, 
                        dtype='int16')
    
    print(f"Dataset successfully saved to {output_path}")
    # Print size of file in MB
    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    print(f"File size: {file_size_mb:.2f} MB")

if __name__ == "__main__":
    # Default parameters
    n = 10
    seed = 42
    output_dir = None  # Use the default path resolution in the function
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    if len(sys.argv) > 2:
        seed = int(sys.argv[2])
    if len(sys.argv) > 3:
        output_dir = os.path.abspath(sys.argv[3])  # Convert to absolute path
    
    generate_gauss_dataset(n, seed, output_dir)