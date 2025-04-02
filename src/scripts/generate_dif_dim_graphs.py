"""
Graph Generation Script for Different Dimensionalities

This script loads benchmark CSV results and creates scatterplots that compare
the performance of different geometric data structures (KDTree, BallTree, BruteForce)
across datasets of varying dimensions. The graphs are generated for various operations such as
k-NN, insertion, deletion, construction, nearest neighbor search, and memory usage.
Each scatterplot is saved as a PNG file.

Key features:
- Loads CSV results into a nested dictionary.
- Creates scatterplots with tailored markers, connecting lines, error bars, and text annotations.
- Sets axis labels and titles according to the operation.
- Saves the graphs in a designated directory.
"""

try:
    import os
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    print("All imports successful")
except ImportError as e:
    print(f"Import error: {e}")
    exit(1)

# Define constants
OPERATIONS = ["get_knn", "insert", "construction", "nearest", "space", "delete"]
DATASETS = ["sift", "enron", "trevi"]
DATASETS_TITLES = ["sift (128d)", "enron (1,369d)", "trevi (4,096d)"]
DATA_STRUCTURES = ["KDTree", "BallTree", "BruteForce"]
DATA_STRUCTURES_LABELS = ["KD Tree", "Ball* Tree", "Brute Force"]
MARKERS = ['o', 's', '^']  # Circle, Square, Triangle markers
COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green colors

def format_bytes(bytes_value):
    """
    Format a byte value into a human-readable string with appropriate unit.

    Args:
        bytes_value (int): The number of bytes.
    
    Returns:
        str: Human-readable format (B, KB, MB, or GB).
    """
    if bytes_value >= 1_000_000_000:  # 1GB or more
        return f"{bytes_value/1_000_000_000:.2f}GB"
    elif bytes_value >= 1_000_000:    # 1MB or more
        return f"{bytes_value/1_000_000:.2f}MB"
    elif bytes_value >= 1_000:        # 1KB or more
        return f"{bytes_value/1_000:.2f}KB"
    else:
        return f"{bytes_value:.0f}B"

def load_results(results_dir):
    """
    Load benchmark result data from CSV files into a nested dictionary.

    The function iterates over each combination of operation, dataset, and 
    data structure. If a file exists, it extracts the relevant data (average time 
    or memory used) and calculates standard deviation (if applicable).

    Args:
        results_dir (str): Directory where benchmark CSV files are stored.
    
    Returns:
        dict: Nested dictionary structured as [operation][dataset][data_structure]
              with keys 'mean' and 'std' for each result.
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
                        # For memory usage, we only need the mean; std dev is set to zero.
                        value = df['space_used'].mean()
                        results[operation][dataset][data_structure] = {'mean': value, 'std': 0}
                    else:
                        # For time-based operations, use avg_time and calculate std dev.
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
    Create and save a scatterplot for a specific benchmark operation comparing performance on different dimensional datasets.

    The function plots the mean performance metrics for each data structure with error bars.
    Data points are slightly offset to prevent marker overlap, and each point is annotated with its value.
    The finished scatterplot is saved as a PNG file.

    Args:
        operation (str): Name of the benchmark operation.
        results (dict): Nested dictionary with benchmark results.
        output_dir (str): Directory where the generated graph will be saved.
    """
    plt.figure(figsize=(10, 7))
    
    # Create x positions corresponding to each dataset.
    x_positions = np.arange(len(DATASETS))
    
    # Determine the maximum mean value across all data structures for scaling annotations.
    max_val = 0
    for dataset in DATASETS:
        for data_structure in DATA_STRUCTURES:
            mean_val = results[operation][dataset][data_structure]['mean']
            if not np.isnan(mean_val):
                max_val = max(max_val, mean_val)
    
    # Plot data for each data structure with a slight x-offset.
    for i, data_structure in enumerate(DATA_STRUCTURES):
        # Offset x positions to avoid overlapping markers.
        offset = (i - 1) * 0.15
        x_positions_offset = x_positions + offset

        # Retrieve y-values and corresponding error values from the results.
        y_values = [results[operation][dataset][data_structure]['mean'] for dataset in DATASETS]
        y_errors = [results[operation][dataset][data_structure]['std'] for dataset in DATASETS]

        # Scatter plot: draw individual data points.
        plt.scatter(x_positions_offset, y_values, 
                    s=100,                   # Marker size.
                    marker=MARKERS[i],
                    color=COLORS[i],
                    label=DATA_STRUCTURES_LABELS[i],
                    alpha=0.8)
        
        # Plot connecting line for visual continuity.
        plt.plot(x_positions_offset, y_values, 
                 color=COLORS[i], 
                 alpha=0.5)
        
        # Add error bars for operations, except construction and space.
        if operation not in ["construction", "space"]:
            plt.errorbar(x_positions_offset, y_values, yerr=y_errors, 
                         fmt='none', ecolor=COLORS[i], capsize=5, alpha=0.6)
        
        # Annotate the data points with their values.
        for j, val in enumerate(y_values):
            if not np.isnan(val):
                # Format the label text based on the type of operation.
                if operation == "space":
                    label_text = format_bytes(val)
                elif operation == "construction":
                    label_text = f'{val:.2f}'
                else:
                    label_text = f'{val:.5f}'
                
                # Compute vertical offset for annotations to avoid overlapping.
                base_offset = max_val * 0.03  # Base offset relative to the maximum y-value.
                stagger_offset = max_val * 0.03 * i  # Additional offset per data structure.
                vertical_position = val + base_offset + stagger_offset
                
                # Place the text annotation.
                plt.text(x_positions_offset[j], vertical_position, 
                         label_text,
                         ha='center', va='bottom',
                         fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))
    
    # Set plot labels and title.
    plt.xlabel('Dataset', fontsize=14)
    if operation == "space":
        plt.ylabel('Memory Usage (bytes)', fontsize=14)
    else:
        plt.ylabel('Average Time (seconds)', fontsize=14)
    plt.title(f'Performance Comparison - {operation.upper()}', fontsize=16, y=1.05)
    plt.xticks(x_positions, DATASETS_TITLES)
    
    # Enhance plot readability by adding grid and legend.
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    
    # Save and close the generated scatterplot.
    output_path = os.path.join(output_dir, f"{operation}_comparison.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved scatterplot to {output_path}")

def main():
    """
    Main function to generate performance graphs across different dimensional datasets.

    It sets up the required directory paths, loads benchmark results,
    iterates over different benchmark operations, and generates corresponding graphs.
    """
    # Determine directory paths for results and graph output.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.dirname(script_dir)
    results_dir = os.path.join(src_dir, "results", "Final_Results")
    output_dir = os.path.join(src_dir, "graphs", "dif_dim_graphs")
    
    # Create the output directory if it does not exist.
    os.makedirs(output_dir, exist_ok=True)
    
    # Load benchmark results.
    print("Loading results...")
    results = load_results(results_dir)
    
    # Generate a graph for each defined operation.
    print("Generating scatterplots...")
    for operation in OPERATIONS:
        create_graph(operation, results, output_dir)
    
    print(f"Done! All scatterplots have been generated in {output_dir}")

if __name__ == "__main__":
    main()