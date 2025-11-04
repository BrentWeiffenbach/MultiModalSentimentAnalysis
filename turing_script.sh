#!/bin/bash
#SBATCH -N 1                   #
#SBATCH -n 8                   #
#SBATCH --mem=8g                        #
#SBATCH -J "MultiModalSentimentAnalysis"   #
#SBATCH -p short               #
#SBATCH -t 12:00:00            #
#SBATCH --gres=gpu:2           #

module load python    #
module load cuda/12.2 #

python dataset_mvsa.py #
