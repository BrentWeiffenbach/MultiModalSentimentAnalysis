#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "bert_pretraining"
#SBATCH -o bert_pretraining%j.out
#SBATCH -e bert_pretraining%j.err
#SBATCH -p academic
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module load python/3.10.12
module load miniconda3
module load cuda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmsa_env

python train_bert.py