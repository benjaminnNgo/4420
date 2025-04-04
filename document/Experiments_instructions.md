# Experiments Running

Before running experiments, ensure that you have activated the virtual environment with the required libraries installed.  
For example, on macOS run:

```bash
source 4420_project_virtual_env/bin/activate
```

We used the following virtual environment on gallium-01 on aviary (would not recommend running this on labtop). Here is the virtual environment we used there:
CONDA_ENV = myenv
CONDA_PREFIX = /data/ugrad/rostj/miniforge

### Running Unit Tests

To run the unit tests:

```bash
make unit_tests
```

### Running Benchmark Tests

To run a single benchmark test, use:

```bash
make benchmark_tests data_structure=<data_structure> dataset=<dataset> operation=<operation> k=<k> trials=<trials> num_operations=<num_operations> radius=<radius> [use_max_queries=true] [cap_dataset=true]
```

- **data_structure**: Determines whether Ball Tree, KD Tree, or Brute Force is run.
- **dataset**: Specifies the dataset (e.g., `cifar`).
- **operation**: Specifies the operation to benchmark (e.g., `space` or `insert`).
- **k**: Number of neighbors for k-NN (ignored for other operations).
- **trials**: Number of times each query is repeated.
- **num_operations**: Number of queries to perform (maximum 200).
- **radius**: Radius for range search (implementation partially complete).
- **use_max_queries**: If set to `true`, overrides `num_operations` to use the maximum allowed.
- **cap_dataset**: If set to `true`, caps the dataset at 50,000 points (useful for dimensionality testing).

See make help for examples

```bash
make help
```

### Running Suites of Benchmark Tests

- **Dimensionality Tests:**

  - Run a suite with a specific data structure:
    ```bash
    make run_{tree name}_dif_dim
    ```
    For example, to run dimensionality tests for the KDTree:
    ```bash
    make run_kdtree_dif_dim
    ```

- **Size Tests:**

  - To run all size tests for different configurations:
    ```bash
    make run_dif_size
    ```

- **Dimensionality Tests (All):**
  - To run all dimensionality benchmark tests:
    ```bash
    make run_dif_dim
    ```
