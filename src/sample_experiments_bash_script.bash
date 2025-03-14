#to run this bash script, use this command: bash sample_experiments_bash_script.bash 

#Running first experiment
python scripts/running_experiment_sample.py --data_structure=kdtree --dataset=cifar --operation=insert

# When previous experiments done, do this experiment next:
python scripts/running_experiment_sample.py --data_structure=kdtree --dataset=cifar --operation=delete

#Then
python scripts/running_experiment_sample.py --data_structure=kdtree --dataset=cifar --operation=get_nearst_neighbour

# And so on...
