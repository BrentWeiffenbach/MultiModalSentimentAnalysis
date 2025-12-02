#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "mvsa_single_inference"
#SBATCH -o mvsa_single_inference%j.out
#SBATCH -e mvsa_single_inference%j.err
#SBATCH -p academic
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module load python/3.10.12
module load miniconda3
module load cuda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmsa_env

python mvsa_single_inference.py --model-path ./experiments/final_training/full_training_custom_finetune/best_model.pth
