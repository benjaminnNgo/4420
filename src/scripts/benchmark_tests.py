"""
Benchmark Tests Script

This script runs a series of benchmarks on various geometric data structures
(KDTree, BallTree, BruteForce) using different datasets and operations.
The benchmarks measure performance metrics such as construction time,
query time (k-NN, nearest neighbor, range search), insertion, deletion, and
memory usage. Results are saved to CSV files for later analysis.

Usage:
    Run this script from the command-line with appropriate arguments.
    Example:
        python benchmark_tests.py --dataset cifar --operation insert --data_structure KDTree
"""

import csv
import copy
import numpy as np
import sys
import os
import argparse

# Adjust path to import modules from the parent directory.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Constants for available datasets, operations, and data structures.
DATASETS = [
    "cifar", "audio", "deep", "enron", "glove", "imageNet",
    "millionSong", "MNIST", "notre", "nuswide", "sift",
    "sun", "trevi", "ukbench", "gauss"
]
DATA_STRUCTURES = ["KDTree", "BallTree", "BruteForce"]
OPERATIONS = ["get_knn", "insert", "delete", "construction", "nearest", "space", "range_search"]
OPERATIONS_NO_SPACE = ["get_knn", "insert", "delete", "construction", "nearest"]

# Import loaders and performance measurement functions from our modules.
from data_loader.data_utils import data_loader  
from tree import KDTree, BallTree, BruteForce               
from tree.utils import (
    measure_get_knn_performance, 
    measure_insert_performance, 
    measure_delete_performance, 
    measure_get_nearest_performance, 
    measure_space_usage, 
    measure_range_search_performance  
)

################################################################################
# Helper Functions
################################################################################
def run_space_benchmark(data_name, data_structure="KDTree", cap_dataset=False):
    """
    Run space benchmark for a given data structure.
    
    The function loads a dataset, constructs the specified data structure,
    measures its memory usage, and returns the result in a list format.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list containing a dictionary with the space used by the data structure.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points

    # Instantiate the chosen data structure.
    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    # Measure the memory usage.
    space = measure_space_usage(tree)
    return [{
        'tree': data_structure,
        'data_name': data_name,
        'space_used': space
    }]

def run_knn_benchmark(data_name, data_structure="KDTree", k=5, trials=5, num_queries=5, use_max_queries=False, cap_dataset=False):
    """
    Run k-NN (k-Nearest Neighbor) benchmark for a given data structure and dataset.
    
    Loads the dataset, builds the specified data structure, and then performs
    multiple k-NN queries for timing measurement. Each query is run for a number
    of trials, and the average query time is recorded.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        k (int): Number of nearest neighbors to query.
        trials (int): Number of trials to run for each query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, uses all available queries.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list of dictionaries containing timed results for each query.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points
    queries = data.queries

    # Determine the number of queries to perform.
    if use_max_queries:
        num_queries = len(queries)
    else:
        num_queries = min(num_queries, len(queries))

    # Instantiate the chosen data structure.
    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    # Execute benchmark per query.
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        
        # Run multiple trials for the same query.
        trial_times = []
        for _ in range(trials):
            elapsed_time = measure_get_knn_performance(tree, target_query, k)
            trial_times.append(elapsed_time)
        
        # Prepare result dictionary with average and individual trial times.
        result = {
            'tree': data_structure,
            'data_name': data_name,
            'query_idx': query_idx,
            'avg_time': sum(trial_times) / trials
        }
        for i, time in enumerate(trial_times):
            result[f'trial_{i+1}'] = time
        
        all_results.append(result)
    return all_results

def run_nearest_benchmark(data_name, data_structure="KDTree", trials=5, num_queries=5, use_max_queries=False, cap_dataset=False):
    """
    Run nearest neighbor benchmark for a given data structure and dataset.
    
    Loads the dataset, builds the specified data structure, and then performs
    nearest neighbor queries, measuring the elapsed time for each.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure.
        trials (int): Number of trials per query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, uses the maximum number of queries.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list of benchmark results for nearest neighbor queries.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points
    queries = data.queries

    if use_max_queries:
        num_queries = len(queries)
    else:
        num_queries = min(num_queries, len(queries))
    
    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        
        trial_times = []
        for _ in range(trials):
            elapsed_time = measure_get_nearest_performance(tree, target_query)
            trial_times.append(elapsed_time)
        
        result = {
            'tree': data_structure,
            'data_name': data_name,
            'query_idx': query_idx,
            'avg_time': sum(trial_times) / trials
        }
        for i, time in enumerate(trial_times):
            result[f'trial_{i+1}'] = time
        
        all_results.append(result)
    return all_results

def run_insert_benchmark(data_name, data_structure="KDTree", num_inserts=5, trials=5, use_max_queries=False, cap_dataset=False):
    """
    Run insert benchmark for a given data structure and dataset.
    
    Constructs the data structure and then benchmarks the insertion operation.
    Uses deep copies of the tree for each trial to ensure consistency.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure.
        num_inserts (int): Number of insertion operations to benchmark.
        trials (int): Number of trials per insertion.
        use_max_queries (bool): If True, uses the maximum number of insert operations.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list of dictionaries with results of the insert benchmark.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points
    queries = data.queries

    if use_max_queries:
        num_inserts = len(queries)
    else:
        num_inserts = min(num_inserts, len(queries))

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for insert_idx in range(num_inserts):
        # For insert benchmarks, a query point is used as the new point.
        new_point = np.array(queries[insert_idx])
        
        trial_times = []
        for _ in range(trials):
            tree_copy = copy.deepcopy(tree)
            elapsed_time = measure_insert_performance(tree_copy, new_point)
            trial_times.append(elapsed_time)
        
        result = {
            'tree': data_structure,
            'data_name': data_name,
            'query_idx': insert_idx,
            'avg_time': sum(trial_times) / trials
        }
        for i, time in enumerate(trial_times):
            result[f'trial_{i+1}'] = time
        
        all_results.append(result)
    return all_results

def run_delete_benchmark(data_name, data_structure="KDTree", num_deletes=5, trials=5, use_max_queries=False, cap_dataset=False):
    """
    Run delete benchmark for a given data structure and dataset.
    
    Constructs the data structure, then iteratively deletes points while
    performing multiple trials for each deletion. Uses deep copies to avoid
    cumulative modifications.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure.
        num_deletes (int): Number of deletions to benchmark.
        trials (int): Number of trials per deletion.
        use_max_queries (bool): If True, uses the maximum number of deletions.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list of dictionaries with deletion benchmark results.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points
    queries = data.queries

    if use_max_queries:
        num_deletes = len(queries)
    else:
        num_deletes = min(num_deletes, len(queries))

    all_results = []
    deleted_points = set()
    
    # Create initial tree with all points.
    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)
    
    # Loop through dataset and delete points that have not yet been deleted.
    for delete_idx in range(min(num_deletes, len(dataset))):
        target_point = None
        for point in dataset:
            if tuple(point) not in deleted_points:
                target_point = point
                deleted_points.add(tuple(point))
                break

        if target_point is None:
            break  # No more points to delete
        
        trial_times = []
        for _ in range(trials):
            tree_copy = copy.deepcopy(tree)
            elapsed_time = measure_delete_performance(tree_copy, target_point)
            trial_times.append(elapsed_time)
        
        result = {
            'tree': data_structure,
            'data_name': data_name,
            'delete_idx': delete_idx,
            'avg_time': sum(trial_times) / trials
        }
        for i, time in enumerate(trial_times):
            result[f'trial_{i+1}'] = time
        
        all_results.append(result)
    return all_results

def run_range_search_benchmark(data_name, data_structure="KDTree", radius=0.5, trials=5, num_queries=5, use_max_queries=False, cap_dataset=False):
    """
    Run range search benchmark for a given data structure and dataset.
    
    Constructs the selected data structure and benchmarks the range search
    operation, measuring the elapsed time for each query. Query results are used
    only for timing; the correctness of the results is assumed.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure.
        radius (float): The search radius.
        trials (int): Number of trials per query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, uses all available queries.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list of dictionaries with range search benchmark results.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points
    queries = data.queries

    if use_max_queries:
        num_queries = len(queries)
    else:
        num_queries = min(num_queries, len(queries))
    
    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        
        trial_times = []
        for _ in range(trials):
            elapsed_time = measure_range_search_performance(tree, target_query, radius)
            trial_times.append(elapsed_time)
        
        result = {
            'tree': data_structure,
            'data_name': data_name,
            'query_idx': query_idx,
            'radius': radius,
            'avg_time': sum(trial_times) / trials
        }
        for i, time in enumerate(trial_times):
            result[f'trial_{i+1}'] = time
        
        all_results.append(result)
    return all_results

def run_construction_benchmark(data_name, data_structure="KDTree", trials=5, cap_dataset=False):
    """
    Run construction benchmark for a given data structure and dataset.
    
    Measures the time required to construct the data structure from the dataset
    over multiple trials. This benchmark provides insight into the construction
    overhead of different spatial data structures.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure.
        trials (int): Number of construction trials.
        cap_dataset (bool): If True, caps the dataset size.
    
    Returns:
        list: A list with a single dictionary containing average construction time.
    """
    data = data_loader(data_name, cap_dataset=cap_dataset)
    dataset = data.points

    import timeit

    trial_times = []
    for _ in range(trials):
        start = timeit.default_timer()
        if data_structure == "KDTree":
            tree = KDTree(dimension=data.dim, points=dataset)
        elif data_structure == "BallTree":
            tree = BallTree(dimension=data.dim, points=dataset)
        elif data_structure == "BruteForce":
            tree = BruteForce(dimension=data.dim, points=dataset)
        elapsed_time = timeit.default_timer() - start
        trial_times.append(elapsed_time)

    result = {
        'tree': data_structure,
        'data_name': data_name,
        'operation': 'construction',
        'avg_time': sum(trial_times) / trials
    }
    for i, time in enumerate(trial_times):
        result[f'trial_{i+1}'] = time
    return [result]

def save_results_csv(results, filename):
    """
    Save the benchmark results to a CSV file.
    
    The function gathers all result keys to ensure that all columns are written,
    then writes a header row followed by one row per result.
    
    Args:
        results (list): List of dictionaries with benchmark results.
        filename (str): Path to the CSV file to save.
    """
    # Determine all unique keys used across results.
    all_keys = set()
    for result in results:
        all_keys.update(result.keys())
    keys = sorted(list(all_keys))
    
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def benchmark_and_save(data_name, operation="get_knn", data_structure="KDTree", k=5, trials=5, num_operations=5, radius=3.0, use_max_queries=False, cap_dataset=False):
    """
    Run the specified benchmark and save the results to a CSV file.
    
    This function selects the benchmark function based on the operation argument,
    runs the benchmark, and then saves the results in a standardized CSV file format.
    
    Args:
        data_name (str): Dataset to test.
        operation (str): Operation to benchmark ("get_knn", "insert", "delete", "construction", "nearest", "space", "range_search").
        data_structure (str): Geometric data structure to test.
        k (int): k parameter for k-NN operations.
        trials (int): Number of trials.
        num_operations (int): Number of operations to perform.
        radius (float): Radius for range search.
        use_max_queries (bool): If True, use maximum queries allowed.
        cap_dataset (bool): If True, cap the dataset size.
    """
    if operation == "get_knn":
        results_list = run_knn_benchmark(data_name, data_structure=data_structure, k=k, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries, cap_dataset=cap_dataset)
    elif operation == "insert":
        results_list = run_insert_benchmark(data_name, data_structure=data_structure, num_inserts=num_operations, trials=trials, use_max_queries=use_max_queries, cap_dataset=cap_dataset)
    elif operation == "delete":
        results_list = run_delete_benchmark(data_name, data_structure=data_structure, num_deletes=num_operations, trials=trials, use_max_queries=use_max_queries, cap_dataset=cap_dataset)
    elif operation == "construction":
        results_list = run_construction_benchmark(data_name, data_structure=data_structure, trials=trials, cap_dataset=cap_dataset)
    elif operation == "nearest":
        results_list = run_nearest_benchmark(data_name, data_structure=data_structure, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries, cap_dataset=cap_dataset)
    elif operation == "space":
        results_list = run_space_benchmark(data_name, data_structure=data_structure, cap_dataset=cap_dataset)
    elif operation == "range_search":
        results_list = run_range_search_benchmark(data_name, data_structure=data_structure, radius=radius, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries, cap_dataset=cap_dataset)
    else:
        raise Exception("Unsupported operation")

    # Determine the directory to save results.
    script_dir = os.path.dirname(__file__)
    src_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(src_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    # Build the output CSV file path.
    csv_file = os.path.join(results_dir, f"{data_name}_{operation}_{data_structure}_benchmark.csv")
    save_results_csv(results_list, csv_file)
    print(f"Benchmark results saved to {csv_file}")

################################################################################
# Command-Line Argument Parsing and Benchmark Execution
################################################################################
parser = argparse.ArgumentParser(description='Parse args for experiments on geometric data structures')
parser.add_argument('--data_structure', type=str, default="KDTree", choices=DATA_STRUCTURES, help='Name of geometric data structure to test')
parser.add_argument('--dataset', type=str, default="cifar", choices=DATASETS, help='Name of dataset to test')
parser.add_argument('--operation', type=str, default="insert", choices=OPERATIONS, help='Name of operation to test')
parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to query (for "get_knn", "nearest" operations)')
parser.add_argument('--trials', type=int, default=5, help='Number of trials to run for each operation')
parser.add_argument('--num_operations', type=int, default=5, help='Number of operations to perform')
parser.add_argument('--radius', type=float, default=3.0, help='Radius for range search queries')
parser.add_argument('--use_max_queries', action='store_true', default=False, help='If True, overrides num_queries and sets it to the max possible value')
parser.add_argument('--cap_dataset', action='store_true', default=False, help='If True, caps the dataset size to 50,000 points')
args = parser.parse_args()

if __name__ == '__main__':
    # Execute benchmark based on provided command-line arguments and save the results.
    benchmark_and_save(
        data_name=args.dataset,
        operation=args.operation,
        data_structure=args.data_structure,
        k=args.k,
        trials=args.trials,
        num_operations=args.num_operations,
        radius=args.radius,
        use_max_queries=args.use_max_queries,
        cap_dataset=args.cap_dataset
    )

    # The following loop is commented out; it can be used for batch benchmarking across all datasets,
    # operations (except "space"), and data structures if needed.
    # for dataset in DATASETS:
    #     for operation in OPERATIONS_NO_SPACE:
    #         for data_structure in DATA_STRUCTURES:
    #             benchmark_and_save(dataset, operation=operation, data_structure=data_structure, k=5, trials=1, num_operations=50, use_max_queries=args.use_max_queries)
