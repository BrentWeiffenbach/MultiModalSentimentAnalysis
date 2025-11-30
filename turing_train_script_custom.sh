#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "full_training_custom_finetune"
#SBATCH -o full_training_custom_finetune%j.out
#SBATCH -e full_training_custom_finetune%j.err
#SBATCH -p academic
#SBATCH -t 48:00:00
#SBATCH --gres=gpu:1

module load python/3.10.12
module load miniconda3
module load cuda

source $(conda info --base)/etc/profile.d/conda.sh
conda activate mmsa_env

python train.py --batch-size 64 --pretrained-vit-path pretrained_encoders/mae_small/best_vit_encoder.pth \
  --pretrained-bert-path pretrained_encoders/bert_medium/best_bert_encoder.pth