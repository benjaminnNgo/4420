# Makefile for running unit tests or benchmark tests

# Python 3 executable
PYTHON3 = python3

# Directories for the tests
SCRIPTS_DIR = src/scripts
UNIT_TESTS = $(SCRIPTS_DIR)/unit_tests.py
BENCHMARK_TESTS = $(SCRIPTS_DIR)/benchmark_tests.py

# Default parameters for benchmark tests
K = 5
TRIALS = 5
NUM_OPERATIONS = 500
RADIUS = 3.0
# USE_MAX_QUERIES = false
USE_MAX_QUERIES = true 

# Options for data structures, datasets, and operations
DATA_STRUCTURES = KDTree BallTree BruteForce
DATASETS = cifar audio deep enron glove imageNet millionSong MNIST notre nuswide sift sun trevi ukbench
DIF_SIZE_DATASETS = cifar sun gauss
# DIF_DIM_DATASETS = 
OPERATIONS = get_knn insert delete construction nearest space range_search

# Default target
.PHONY: all
all: unit_tests

# Run unit tests
.PHONY: unit_tests
unit_tests:
	$(PYTHON3) $(UNIT_TESTS)

# Run benchmark tests with arguments
.PHONY: benchmark_tests
benchmark_tests:
	$(PYTHON3) $(BENCHMARK_TESTS) --data_structure=$(data_structure) --dataset=$(dataset) --operation=$(operation) --k=$(k) --trials=$(trials) --num_operations=$(num_operations) --radius=$(radius) $(if $(filter true,$(use_max_queries)),--use_max_queries)

# Run all combinations of data structures, operations, and datasets
.PHONY: run_all
run_all:
	@for data_structure in $(DATA_STRUCTURES); do \
		for dataset in $(DATASETS); do \
			for operation in $(OPERATIONS); do \
				$(MAKE) benchmark_tests data_structure=$$data_structure dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) ; \
			done; \
		done; \
	done;

# Run all combinations of data structures, operations, and datasets
.PHONY: run_dif_size
run_dif_size:
	@for data_structure in $(DATA_STRUCTURES); do \
		for dataset in $(DIF_SIZE_DATASETS); do \
			for operation in $(OPERATIONS); do \
				$(MAKE) benchmark_tests data_structure=$$data_structure dataset=$$dataset operation=$$operation k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=$(USE_MAX_QUERIES) ; \
			done; \
		done; \
	done;

# Help target to provide the correct format for benchmark_tests
.PHONY: help
help:
	@echo "To run the benchmark tests manually, use the following format:"
	@echo "  make benchmark_tests data_structure=<data_structure> dataset=<dataset> operation=<operation> k=<k> trials=<trials> num_operations=<num_operations> radius=<radius> [use_max_queries=true]"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark_tests data_structure=KDTree dataset=cifar operation=get_knn k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS) use_max_queries=true"
	@echo "  make benchmark_tests data_structure=BallTree dataset=sift operation=insert k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS)"
	@echo "  make benchmark_tests data_structure=BruteForce dataset=MNIST operation=delete k=$(K) trials=$(TRIALS) num_operations=$(NUM_OPERATIONS) radius=$(RADIUS)"
	@echo ""
	@echo "For more details on each parameter, refer to the 'benchmark_tests' section in the Makefile."

# Clean up any generated files (if any)
.PHONY: clean
clean:
	rm -rf results