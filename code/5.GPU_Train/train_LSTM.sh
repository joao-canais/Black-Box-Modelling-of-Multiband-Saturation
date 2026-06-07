#!/bin/bash
#SBATCH --job-name=LSTM
#SBATCH --output=LSTM_output_%j.txt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16gb
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --nodelist=opel

export PROJECT_DIR=$HOME/projeto
export DATASET_DIR=$PROJECT_DIR/data/Dataset
export RESULTS_DIR=$PROJECT_DIR/LSTM_results

cd "$PROJECT_DIR"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate torch-env

python -c "import torch; print('CUDA:', torch.cuda.is_available())"
python train_LSTM.py