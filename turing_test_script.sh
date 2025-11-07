#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 1                      # total number of tasks
#SBATCH --mem=8g                  # allocate 1 GB of memory
#SBATCH -J "dataset_setup"      # name of the job
#SBATCH -o dataset_setup%j.out # name of the output file
#SBATCH -e dataset_setup%j.err # name of the error file
#SBATCH -p academic                  # partition to submit to
#SBATCH -t 01:00:00               # time limit of 1 hour
#SBATCH --gres=gpu:1              # request 1 GPU

module load python/3.10.12
module load miniconda3
module load cuda
module list

# Create the conda environment
conda create -y -n mmsa_env python=3.10.12
source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmsa_env
pip install -r requirements.txt

python3 dataset_mvsa.py