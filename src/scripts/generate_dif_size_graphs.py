"""
Graph Generation Script for Different Dataset Sizes

This script loads benchmark CSV results and creates scatterplots that compare
the performance of different geometric data structures (KDTree, BallTree, BruteForce)
across multiple datasets. The graphs are generated for various operations such as
k-NN, insertion, deletion, construction, nearest neighbor search, memory usage, and
range search. Each scatterplot is saved as a PNG file.

Key functionalities:
- Load result data from CSV files.
- Generate a scatterplot for each benchmark operation.
- Annotate plots with error bars and value labels.
- Save plots to a designated directory.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Constant definitions for operations, datasets, data structures, and styling.
OPERATIONS = ["get_knn", "insert", "construction", "nearest", "space", "delete"]
DATASETS = ["cifar", "sun", "gauss"]
DATASETS_TITLES = ["cifar (50,000)", "sun (79,000)", "gauss (2,000,000)"]
DATA_STRUCTURES = ["KDTree", "BallTree", "BruteForce"]
DATA_STRUCTURES_LABELS = ["KD Tree", "Ball* Tree", "Brute Force"]
MARKERS = ['o', 's', '^']  # Circle, Square, Triangle markers.
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green colors.

def format_bytes(bytes_value):
    """ 
    Convert a byte value into a human-readable format (KB, MB, or GB).

    Args:
        bytes_value (int): The number of bytes.

    Returns:
        str: Formatted string representing the size.
    """
    if bytes_value >= 1_000_000_000:  # 1GB or more.
        return f"{bytes_value/1_000_000_000:.2f}GB"
    elif bytes_value >= 1_000_000:    # 1MB or more.
        return f"{bytes_value/1_000_000:.2f}MB"
    elif bytes_value >= 1_000:        # 1KB or more.
        return f"{bytes_value/1_000:.2f}KB"
    else:
        return f"{bytes_value:.0f}B"

def load_results(results_dir):
    """
    Load benchmark results from CSV files into a nested dictionary.

    The function iterates over all combinations of operations, datasets and
    data structures, loading each CSV file and extracting the relevant columns.
    If an expected file is not found, it issues a warning and fills with NaN.

    Args:
        results_dir (str): Path to the directory containing benchmark CSV files.

    Returns:
        dict: Nested dictionary structure [operation][dataset][data_structure]
              with means and standard deviations.
    """
    results = {}
    
    for operation in OPERATIONS:
        results[operation] = {}
        for dataset in DATASETS:
            results[operation][dataset] = {}
            for data_structure in DATA_STRUCTURES:
                file_path = os.path.join(results_dir, f"{dataset}_{operation}_{data_structure}_benchmark.csv")
                try:
                    df = pd.read_csv(file_path)
                    if operation == "space":
                        # For memory usage, we only need the mean value (std deviation not applicable).
                        value = df['space_used'].mean()
                        results[operation][dataset][data_structure] = {'mean': value, 'std': 0}
                    else:
                        # For timing metrics, use 'avg_time' column and calculate std deviation.
                        mean_value = df['avg_time'].mean()
                        std_value = df['avg_time'].std() if operation not in ["construction", "space"] else 0
                        results[operation][dataset][data_structure] = {'mean': mean_value, 'std': std_value}
                except FileNotFoundError:
                    print(f"Warning: File not found - {file_path}")
                    results[operation][dataset][data_structure] = {'mean': np.nan, 'std': np.nan}
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    results[operation][dataset][data_structure] = {'mean': np.nan, 'std': np.nan}
    
    return results

def create_graph(operation, results, output_dir):
    """
    Create and save a scatterplot for a specified benchmark operation.

    The function plots performance metrics for each data structure over all datasets.
    It adds markers, error bars (when applicable) and text labels to annotate the plot.
    The finished plot is saved as a PNG file.

    Args:
        operation (str): Benchmark operation name.
        results (dict): Nested dictionary with benchmark results.
        output_dir (str): Directory where the generated graph is saved.
    """
    plt.figure(figsize=(10, 7))
    
    # Generate x positions corresponding to the datasets.
    x_positions = np.arange(len(DATASETS))
    
    # Determine maximum y-value across all data structures to aid in positioning labels.
    max_val = 0
    for dataset in DATASETS:
        for data_structure in DATA_STRUCTURES:
            if not np.isnan(results[operation][dataset][data_structure]['mean']):
                max_val = max(max_val, results[operation][dataset][data_structure]['mean'])
    
    # Plot data for each data structure.
    for i, data_structure in enumerate(DATA_STRUCTURES):
        # Offset x positions slightly to prevent overlap between markers.
        offset = (i - 1) * 0.15
        
        y_values = [results[operation][dataset][data_structure]['mean'] for dataset in DATASETS]
        y_errors = [results[operation][dataset][data_structure]['std'] for dataset in DATASETS]
        x_positions_offset = x_positions + offset
        
        # Scatter plot for the data points.
        plt.scatter(x_positions_offset, y_values, 
                   s=100,                  # Marker size.
                   marker=MARKERS[i],
                   color=COLORS[i], 
                   label=DATA_STRUCTURES_LABELS[i],
                   alpha=0.8)
        
        # Add error bars for operations that require uncertainty visualization.
        if operation not in ["construction", "space"]:
            plt.errorbar(x_positions_offset, y_values, yerr=y_errors, 
                         fmt='none', ecolor=COLORS[i], capsize=5, alpha=0.6)
        
        # Draw connecting lines between data points.
        plt.plot(x_positions_offset, y_values, 
                 color=COLORS[i], 
                 alpha=0.5)
        
        # Annotate each point with its value.
        for j, val in enumerate(y_values):
            if not np.isnan(val):
                if operation == "space":
                    label_text = format_bytes(val)
                elif operation == "construction":
                    label_text = f'{val:.2f}'
                else:
                    label_text = f'{val:.5f}'
                
                # Calculate vertical position to avoid overlap.
                base_offset = max_val * 0.03
                stagger_offset = max_val * 0.03 * i
                vertical_position = val + base_offset + stagger_offset
                
                plt.text(x_positions_offset[j], vertical_position, 
                         label_text,
                         ha='center', va='bottom',
                         fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Set axis labels.
    plt.xlabel('Dataset', fontsize=14)
    
    if operation == "space":
        plt.ylabel('Memory Usage (bytes)', fontsize=14)
    else:
        plt.ylabel('Average Time (seconds)', fontsize=14)
    
    # Set title and adjust layout.
    plt.title(f'Performance Comparison - {operation.upper()}', fontsize=16, y=1.05)
    plt.xticks(x_positions, DATASETS_TITLES)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save the scatterplot to a file.
    output_path = os.path.join(output_dir, f"{operation}_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved scatterplot to {output_path}")

def main():
    """
    Main function to generate graphs for different dataset sizes.

    It determines directory paths for input results and output graphs, loads the benchmark results,
    and then iterates through each benchmark operation to create and save corresponding graphs.
    """
    # Determine essential directories.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(src_dir, "results", "Final_Results")
    output_dir = os.path.join(src_dir, "graphs", "dif_size_graphs")
    
    # Create output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load benchmark results.
    print("Loading results...")
    results = load_results(results_dir)
    
    # Generate graphs for all defined operations.
    print("Generating scatterplots...")
    for operation in OPERATIONS:
        create_graph(operation, results, output_dir)
    
    print(f"Done! All scatterplots have been generated in {output_dir}")

if __name__ == "__main__":
    main()