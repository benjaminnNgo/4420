#==============================================================================
# GEOMETRIC DATA STRUCTURES BENCHMARK MAKEFILE
# 
# This makefile provides targets for testing and benchmarking various
# spatial data structures (KD-Tree, Ball*-Tree, Brute Force) on different
# datasets with varying dimensionality and size.
#
# Main targets:
# - unit_tests: Run unit tests
# - run_dif_size: Compare performance across datasets of different sizes
# - run_dif_dim: Compare performance across datasets of different dimensions
#==============================================================================

#==============================================================================
# CONFIGURATION VARIABLES
#==============================================================================
# Python and environment settings
PYTHON3 = python3

CONDA_ENV = myenv
CONDA_PREFIX = /data/ugrad/rostj/miniforge

# Directories for the tests
SCRIPTS_DIR = src/scripts
UNIT_TESTS = $(SCRIPTS_DIR)/unit_tests.py
BENCHMARK_TESTS = $(SCRIPTS_DIR)/benchmark_tests.py

#==============================================================================
# TEST PARAMETERS
#==============================================================================
# Default parameters for benchmark tests
K = 5
TRIALS = 1
NUM_OPERATIONS = 200
RADIUS = 3.0
# USE_MAX_QUERIES = false
USE_MAX_QUERIES = true
CAP_DATASET = true  # Add this line

# Options for data structures, datasets, and operations
DATA_STRUCTURES = KDTree BallTree BruteForce
DATASETS = cifar audio deep enron glove imageNet millionSong MNIST notre nuswide sift sun trevi ukbench
DIF_SIZE_DATASETS = cifar sun gauss
DIF_DIM_DATASETS = sift enron trevi
OPERATIONS = get_knn insert construction nearest space delete

# Default target
.PHONY: all
all: unit_tests

.PHONY: show_pwd
show_pwd:
	@echo "PWD is: $(PWD)"

# Run unit tests
.PHONY: unit_tests
unit_tests:
	$(PYTHON3) $(UNIT_TESTS)

# Run benchmark tests with arguments
.PHONY: benchmark_tests
benchmark_tests:
	$(PYTHON3) $(BENCHMARK_TESTS) --data_structure=$(data_structure) --dataset=$(dataset) --operation=$(operation) --k=$(k) --trials=$(trials) --num_operations=$(num_operations) --radius=$(radius) $(if $(filter true,$(use_max_queries)),--use_max_queries) $(if $(filter true,$(cap_dataset)),--cap_dataset)



# dif size
# KDTree with different datasets
.PHONY: run_kdtree_dif_size
run_kdtree_dif_size:
	@for dataset in $(DIF_SIZE_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=KDTree dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES); \
		done; \
	done;

# BallTree with different datasets
.PHONY: run_balltree_dif_size
run_balltree_dif_size:
	@for dataset in $(DIF_SIZE_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=BallTree dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES); \
		done; \
	done;

# BruteForce with different datasets
.PHONY: run_bruteforce_dif_size
run_bruteforce_dif_size:
	@for dataset in $(DIF_SIZE_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=BruteForce dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES); \
		done; \
	done;



# dif dim
# KDTree with different dimensions
.PHONY: run_kdtree_dif_dim
run_kdtree_dif_dim:
	@for dataset in $(DIF_DIM_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=KDTree dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) cap_dataset=$(CAP_DATASET); \
		done; \
	done;

# BallTree with different dimensions
.PHONY: run_balltree_dif_dim
run_balltree_dif_dim:
	@for dataset in $(DIF_DIM_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=BallTree dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) cap_dataset=$(CAP_DATASET); \
		done; \
	done;

# BruteForce with different dimensions
.PHONY: run_bruteforce_dif_dim
run_bruteforce_dif_dim:
	@for dataset in $(DIF_DIM_DATASETS); do \
		for operation in $(OPERATIONS); do \
			$(MAKE) benchmark_tests data_structure=BruteForce dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) cap_dataset=$(CAP_DATASET); \
		done; \
	done;



# Run all data structures with different size datasets
.PHONY: run_dif_size
run_dif_size:
	@for data_structure in $(DATA_STRUCTURES); do \
		for dataset in $(DIF_SIZE_DATASETS); do \
			for operation in $(OPERATIONS); do \
				$(MAKE) benchmark_tests data_structure=$$data_structure dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) ; \
			done; \
		done; \
	done;

# Run all data structures with different dimension datasets
.PHONY: run_dif_dim
run_dif_dim:
	@for data_structure in $(DATA_STRUCTURES); do \
		for dataset in $(DIF_DIM_DATASETS); do \
			for operation in $(OPERATIONS); do \
				$(MAKE) benchmark_tests data_structure=$$data_structure dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) cap_dataset=$(CAP_DATASET) ; \
			done; \
		done; \
	done;


# Help target to provide the correct format for benchmark_tests
.PHONY: help
help:
	@echo "To run the benchmark tests manually, use the following format:"
	@echo "  make benchmark_tests data_structure=<data_structure> dataset=<dataset> operation=<operation> k=<k> trials=<trials> num_operations=<num_operations> radius=<radius> [use_max_queries=true] [cap_dataset=true]"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark_tests data_structure=KDTree dataset=cifar operation=get_knn k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=true"
	@echo "  make benchmark_tests data_structure=BallTree dataset=sift operation=insert k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) cap_dataset=true"
	@echo "  make benchmark_tests data_structure=BruteForce dataset=MNIST operation=delete k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS)"
	@echo ""
	@echo "For more details on each parameter, refer to the 'benchmark_tests' section in the Makefile."

# Clean up any generated files (if any)
.PHONY: clean
clean:
	rm -rf results