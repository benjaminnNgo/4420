import csv
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data_loader.data_utils import data_loader  # NEW import for data loader
from tree import KDTree, BallTree                  # NEW import for trees
from tree.utils import measure_get_knn_performance  # NEW import for performance measurement


def run_benchmark(data_name, query_idx=0, k=5, trials=5):
    # Load the data (assumes data_loader returns an object with attributes: dim, points, queries)
    data = data_loader(data_name)
    dataset = data.points
    queries = data.queries
    target_query = np.array(queries[query_idx])
    
    # Build the trees
    kd_tree = KDTree(dimension=data.dim, points=dataset)
    ball_tree = BallTree(dimension=data.dim, points=dataset)
    
    kd_time, kd_space = measure_get_knn_performance(kd_tree, target_query, k, trials)
    ball_time, ball_space = measure_get_knn_performance(ball_tree, target_query, k, trials)
    
    kd_result = {
        'tree': 'KDTree',
        'data_name': data_name,
        'elapsed_time': kd_time,
        'space_used': kd_space
    }
    ball_result = {
        'tree': 'BallTree',
        'data_name': data_name,
        'elapsed_time': ball_time,
        'space_used': ball_space
    }
    return [kd_result, ball_result]

def save_results_csv(results, filename):
    keys = results[0].keys()
    with open(filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=keys)
        writer.writeheader()
        for row in results:
            writer.writerow(row)

if __name__ == '__main__':
    data_name = "cifar"
    results_list = run_benchmark(data_name, query_idx=0, k=5, trials=5)
    os.makedirs("results", exist_ok=True)  # Ensure the results folder exists
    csv_file = os.path.join("results", f"{data_name}_benchmark.csv")
    save_results_csv(results_list, csv_file)
    print(f"Benchmark results saved to {csv_file}")