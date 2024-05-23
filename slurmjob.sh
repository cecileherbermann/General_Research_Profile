#!/bin/bash
#SBATCH --job-name=finetune_scGPT  
#SBATCH --output=finetune_scGPT.out
#SBATCH --error=finetune_scGPT.err
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH --mem=80G
#SBATCH --partition=gpu-medium
#SBATCH --gpus-per-node=1

# Load the conda environment
conda activate scGPT

# Change directory to scGPT directory
cd "/projects/0/prjs1045"

# Paths
model_path = "/scGPT_human"
raw_data_path = "/CFS_data/CFS_all_days_rawcount.h5ad"
processed_data_path = "CFS_data/CFS_all_days_processed.h5ad"
day = "CFS_Day28"

# Run the script
python finetuning_slurmjob.py --model_path $model_path --raw_data_path $raw_data_path --processed_data_path $processed_data_path --day $day
