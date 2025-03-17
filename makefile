# Makefile for running unit tests or benchmark tests

# Python 3 executable
PYTHON3 = python3

# Directories for the tests
SCRIPTS_DIR = src/scripts
UNIT_TESTS = $(SCRIPTS_DIR)/unit_tests.py
BENCHMARK_TESTS = $(SCRIPTS_DIR)/benchmark_tests.py

# Options for data structures, datasets, and operations
DATA_STRUCTURES = KDTree BallTree BruteForce
DATASETS = cifar audio deep enron glove imageNet millionSong MNIST notre nuswide sift sun trevi ukbench
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
	$(PYTHON3) $(BENCHMARK_TESTS) --data_structure=$(data_structure) --dataset=$(dataset) --operation=$(operation) --k=$(k) --trials=$(trials) --num_operations=$(num_operations) --radius=$(radius) $(if $(use_max_queries),--use_max_queries) 

# Run all combinations of data structures, operations, and datasets
.PHONY: run_all
run_all:
	@for data_structure in $(DATA_STRUCTURES); do \
		for dataset in $(DATASETS); do \
			for operation in $(OPERATIONS); do \
				$(MAKE) benchmark_tests data_structure=$$data_structure dataset=$$dataset operation=$$operation k=5 trials=1 num_operations=50 radius=0.5 --use_max_queries ; \
			done; \
		done; \
	done;

# Help target to provide the correct format for benchmark_tests
.PHONY: help
help:
	@echo "To run the benchmark tests manually, use the following format:"
	@echo "  make benchmark_tests data_structure=<data_structure> dataset=<dataset> operation=<operation> k=<k> trials=<trials> num_operations=<num_operations> radius=<radius> [--use_max_queries]"
	@echo ""
	@echo "Examples:"
	@echo "  make benchmark_tests data_structure=KDTree dataset=cifar operation=get_knn k=5 trials=1 num_operations=50 radius=0.5 --use_max_queries"
	@echo "  make benchmark_tests data_structure=BallTree dataset=sift operation=insert k=5 trials=1 num_operations=100 radius=0.5"
	@echo "  make benchmark_tests data_structure=BruteForce dataset=MNIST operation=delete k=5 trials=1 num_operations=50 radius=0.5"
	@echo ""
	@echo "For more details on each parameter, refer to the 'benchmark_tests' section in the Makefile."

# Clean up any generated files (if any)
.PHONY: clean
clean:
	rm -rf results