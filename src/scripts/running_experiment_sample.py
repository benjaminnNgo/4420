import argparse

parser = argparse.ArgumentParser(description='Parse agrs for experiments on geometric data structure')

parser.add_argument('--data_structure', type=str, default="kdtree", help='Name of geometric data structure to test')
parser.add_argument('--dataset', type=str, default="cifar", help='Name of dataset to test')
parser.add_argument('--operation', type=str, default="insert", help='Name of operation to test')

args = parser.parse_args()


if __name__ == "__main__":
    # python command to run this script:
    # python scripts/running_experiment_sample.py --data_structure=kdtree --dataset=cifar --operation=insert
    print(f"Running experiments on {args.data_structure}, {args.dataset}, {args.operation}")
