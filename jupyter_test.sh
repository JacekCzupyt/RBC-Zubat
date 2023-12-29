#!/bin/bash
#SBATCH --nodes 1
#SBATCH --cpus-per-task 5
#SBATCH --time 03:00:00
#SBATCH --job-name jupyter-notebook

# get tunneling info

port=4976
node=$(hostname -s)
user=$(whoami)

# run jupyter notebook
jupyter-notebook --no-browser --port=${port} --ip=${node}