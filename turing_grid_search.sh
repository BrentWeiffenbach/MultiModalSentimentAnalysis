#!/bin/bash
# Grid search launcher - submits N parallel jobs

NUM_JOBS=${1:-2}  # Default to 2 if not provided

for job in $(seq 1 $NUM_JOBS); do
    sbatch <<EOF
#!/bin/bash
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=32g
#SBATCH -J "train_search_job${job}_"
#SBATCH -o train_search_job${job}_%j.out
#SBATCH -e train_search_job${job}_%j.err
#SBATCH -p academic
#SBATCH -t 24:00:00
#SBATCH --gres=gpu:1

module load python/3.10.12
module load miniconda3
module load cuda

source \$(conda info --base)/etc/profile.d/conda.sh
conda activate mmsa_env
python grid_search.py --grid-search ${job}
EOF
    echo "Submitted job ${job}"
done

echo "All grid search jobs submitted!"
