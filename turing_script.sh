#!/bin/bash
#SBATCH -N 1                      # allocate 1 compute node
#SBATCH -n 1                      # total number of tasks
#SBATCH --mem=1g                  # allocate 1 GB of memory
#SBATCH -J "pytorch example"      # name of the job
#SBATCH -o pytorch_example_%j.out # name of the output file
#SBATCH -e pytorch_example_%j.err # name of the error file
#SBATCH -p short                  # partition to submit to
#SBATCH -t 01:00:00               # time limit of 1 hour
#SBATCH --gres=gpu:1              # request 1 GPU

module load python/3.11.10               # These version were chosen for compatability with pytorch
module load cuda/12.4.0/3mdaov5          # load CUDA (adjust if necessary)

python3 -m venv .venv         #
source .venv/bin/activate     #
pip install -r requirements.txt #
python3 dataset_mvsa.py #
