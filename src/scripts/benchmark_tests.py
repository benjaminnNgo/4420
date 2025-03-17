import csv
import numpy as np
import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

DATASETS = [
    "cifar", "audio", "deep", "enron", "glove", "imageNet",
    "millionSong", "MNIST", "notre", "nuswide", "sift",
    "sun", "trevi", "ukbench"
]
DATA_STRUCTURES = ["KDTree", "BallTree", "BruteForce"]
OPERATIONS = ["get_knn", "insert", "delete", "construction", "nearest", "space", "range_search"] 
OPERATIONS_NO_SPACE = ["get_knn", "insert", "delete", "construction", "nearest"] 

from data_loader.data_utils import data_loader  
from tree import KDTree, BallTree, BruteForce               
from tree.utils import measure_get_knn_performance, measure_insert_performance, measure_delete_performance, measure_get_nearest_performance, measure_space_usage, measure_range_search_performance  

"""
    Run space benchmark for a given data structure.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
    
    Returns:
        list: A list containing the space used by the data structure.
"""
def run_space_benchmark(data_name, data_structure="KDTree"):
    data = data_loader(data_name)
    dataset = data.points[:500]

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    space = measure_space_usage(tree)
    return [{
        'tree': data_structure,
        'data_name': data_name,
        'space_used': space
    }]

"""
    Run k-NN (k-Nearest Neighbor) benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        k (int): Number of nearest neighbors to query.
        trials (int): Number of trials to run for each query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, sets num_queries to the maximum number of queries.
    
    Returns:
        list: A list of results containing the elapsed time for each query.
"""
def run_knn_benchmark(data_name, data_structure="KDTree", k=5, trials=5, num_queries=5, use_max_queries=False):
    data = data_loader(data_name)
    dataset = data.points[:500]
    queries = data.queries

    if use_max_queries:
        num_queries = len(queries)

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        kd_time = measure_get_knn_performance(tree, target_query, k, trials)

        all_results.append({
            'tree': data_structure,
            'data_name': data_name,
            'elapsed_time': kd_time
        })

    return all_results

"""
    Run nearest neighbor benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        trials (int): Number of trials to run for each query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, sets num_queries to the maximum number of queries.
    
    Returns:
        list: A list of results containing the elapsed time for each query.
"""
def run_nearest_benchmark(data_name, data_structure="KDTree", trials=5, num_queries=5, use_max_queries=False):
    data = data_loader(data_name)
    dataset = data.points[:500]
    queries = data.queries

    if use_max_queries:
        num_queries = len(queries)

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        kd_time = measure_get_nearest_performance(tree, target_query, trials)

        all_results.append({
            'tree': data_structure,
            'data_name': data_name,
            'elapsed_time': kd_time
        })

    return all_results

"""
    Run insert benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        num_inserts (int): Number of insertions to perform.
        trials (int): Number of trials to run for each insertion.
    
    Returns:
        list: A list of results containing the elapsed time for each insertion.
"""
def run_insert_benchmark(data_name, data_structure="KDTree", num_inserts=5, trials=5):
    data = data_loader(data_name)
    dataset = data.points[:500]

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for _ in range(num_inserts):
        new_point = np.random.rand(data.dim)
        kd_time = measure_insert_performance(tree, new_point, trials=trials)

        all_results.append({
            'tree': data_structure,
            'data_name': data_name,
            'elapsed_time': kd_time
        })

    return all_results

"""
    Run delete benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        num_deletes (int): Number of deletions to perform.
        trials (int): Number of trials to run for each deletion.
        use_max_queries (bool): If True, sets num_deletes to the maximum number of deletions.
    
    Returns:
        list: A list of results containing the elapsed time for each deletion.
"""
def run_delete_benchmark(data_name, data_structure="KDTree", num_deletes=5, trials=5, use_max_queries=False):
    data = data_loader(data_name)
    dataset = data.points[:500]

    if use_max_queries:
        num_deletes = len(dataset)

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    deleted_points = set()
    all_results = []

    for _ in range(min(num_deletes, len(dataset))):
        target_point = None
        for point in dataset:
            if tuple(point) not in deleted_points:
                target_point = point
                deleted_points.add(tuple(point))
                break

        if target_point is None:
            break  # No more points to delete

        kd_time = measure_delete_performance(tree, target_point, trials=trials)

        all_results.append({
            'tree': data_structure,
            'data_name': data_name,
            'elapsed_time': kd_time
        })

    return all_results

"""
    Run range search benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        radius (float): The radius within which to search for points.
        trials (int): Number of trials to run for each query.
        num_queries (int): Number of queries to run.
        use_max_queries (bool): If True, sets num_queries to the maximum number of queries.
    
    Returns:
        list: A list of results containing the elapsed time for each query.
"""
def run_range_search_benchmark(data_name, data_structure="KDTree", radius=0.5, trials=5, num_queries=0, use_max_queries=False):
    data = data_loader(data_name)
    dataset = data.points[:500]
    queries = data.queries

    if use_max_queries:
        num_queries = len(queries)

    if data_structure == "KDTree":
        tree = KDTree(dimension=data.dim, points=dataset)
    elif data_structure == "BallTree":
        tree = BallTree(dimension=data.dim, points=dataset)
    elif data_structure == "BruteForce":
        tree = BruteForce(dimension=data.dim, points=dataset)

    all_results = []
    for query_idx in range(num_queries):
        target_query = np.array(queries[query_idx])
        range_search_time = measure_range_search_performance(tree, target_query, radius, trials)

        all_results.append({
            'tree': data_structure,
            'data_name': data_name,
            'elapsed_time': range_search_time
        })

    return all_results

"""
    Run construction benchmark for a given data structure and dataset.
    
    Args:
        data_name (str): Dataset to test.
        data_structure (str): Type of geometric data structure ("KDTree", "BallTree", or "BruteForce").
        trials (int): Number of trials to run for the construction.
    
    Returns:
        list: A list containing the average elapsed time for the construction.
"""
def run_construction_benchmark(data_name, data_structure="KDTree", trials=5):
    data = data_loader(data_name)
    dataset = data.points[:500]

    import timeit

    times = []

    for _ in range(trials):
        start = timeit.default_timer()
        if data_structure == "KDTree":
            tree = KDTree(dimension=data.dim, points=dataset)
        elif data_structure == "BallTree":
            tree = BallTree(dimension=data.dim, points=dataset)
        elif data_structure == "BruteForce":
            tree = BruteForce(dimension=data.dim, points=dataset)
        times.append(timeit.default_timer() - start)

    results = [
        {
            'tree': data_structure,
            'data_name': data_name,
            'operation': 'construction',
            'elapsed_time': sum(times) / trials
        }
    ]

    return results

def save_results_csv(results, filename):
    keys = results[0].keys()
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

def benchmark_and_save(data_name, operation="get_knn", data_structure="KDTree", k=5, trials=5, num_operations=5, radius=0.5, use_max_queries=False):
    if operation == "get_knn":
        results_list = run_knn_benchmark(data_name, data_structure=data_structure, k=k, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries)
    elif operation == "insert":
        results_list = run_insert_benchmark(data_name, data_structure=data_structure, num_inserts=num_operations, trials=trials)
    elif operation == "delete":
        results_list = run_delete_benchmark(data_name, data_structure=data_structure, num_deletes=num_operations, trials=trials, use_max_queries=use_max_queries)
    elif operation == "construction":
        results_list = run_construction_benchmark(data_name, data_structure=data_structure, trials=trials)
    elif operation == "nearest":
        results_list = run_nearest_benchmark(data_name, data_structure=data_structure, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries)
    elif operation == "space":
        results_list = run_space_benchmark(data_name, data_structure=data_structure)
    elif operation == "range_search":
        results_list = run_range_search_benchmark(data_name, data_structure=data_structure, radius=radius, trials=trials, num_queries=num_operations, use_max_queries=use_max_queries)
    else:
        raise Exception("Unsupported operation")

    script_dir = os.path.dirname(__file__)
    src_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(src_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    csv_file = os.path.join(results_dir, f"{data_name}_{operation}_{data_structure}_benchmark.csv")
    save_results_csv(results_list, csv_file)
    print(f"Benchmark results saved to {csv_file}")

# Argument parsing setup
parser = argparse.ArgumentParser(description='Parse args for experiments on geometric data structures')
parser.add_argument('--data_structure', type=str, default="KDTree", choices=DATA_STRUCTURES, help='Name of geometric data structure to test')
parser.add_argument('--dataset', type=str, default="cifar", choices=DATASETS, help='Name of dataset to test')
parser.add_argument('--operation', type=str, default="insert", choices=OPERATIONS, help='Name of operation to test')
parser.add_argument('--k', type=int, default=5, help='Number of nearest neighbors to query (for "get_knn", "nearest" operations)')
parser.add_argument('--trials', type=int, default=5, help='Number of trials to run for each operation')
parser.add_argument('--num_operations', type=int, default=5, help='Number of operations to perform')
parser.add_argument('--radius', type=float, default=0.5, help='Radius for range search queries')
parser.add_argument('--use_max_queries', action='store_true', default=False, help='If True, overrides num_queries and sets it to the max possible value')
args = parser.parse_args()

if __name__ == '__main__':
    benchmark_and_save(
        data_name=args.dataset,
        operation=args.operation,
        data_structure=args.data_structure,
        k=args.k,
        trials=args.trials,
        num_operations=args.num_operations,
        radius=args.radius,
        use_max_queries=args.use_max_queries
    )

    # for dataset in DATASETS:
    #     for operation in OPERATIONS_NO_SPACE:
    #         for data_structure in DATA_STRUCTURES:
    #             benchmark_and_save(dataset, operation=operation, data_structure=data_structure, k=5, trials=1, num_operations=50, use_max_queries=args.use_max_queries)
