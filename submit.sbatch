#!/bin/sh
#SBATCH --job-name=gpn_ft
#SBATCH --time=1:00:00
#SBATCH --partition=gpu
#SBATCH --ntasks-per-node=1
#SBATCH --mem=50G
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --account=pi-yangili1
#SBATCH -o log

module load cuda/12.2

# run NT transformer example code
python NT_transformer_example.py
# run segment NT example code
python segNT_example.py
