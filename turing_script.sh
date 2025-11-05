#!/bin/bash
#SBATCH -N 1                   #
#SBATCH -n 8                   #
#SBATCH --mem=8g                        #
#SBATCH -J "MultiModalSentimentAnalysis"   #
#SBATCH -p academic               #
#SBATCH -t 12:00:00            #
#SBATCH --gres=gpu:2           #

module load python/3.10.12    #
module load cuda/12.4.0/3mdaov5 #
python3 -m venv .venv         #
source .venv/bin/activate     #
pip install -r requirements.txt #
python3 dataset_mvsa.py #
