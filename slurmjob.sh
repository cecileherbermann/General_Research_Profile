#!/bin/bash
#SBATCH --job-name=finetune_scGPT  
#SBATCH --output=finetune_scGPT.out
#SBATCH --error=finetune_scGPT.err
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --partition=short

# Load the conda environment
conda activate scGPT

# Change directory to scGPT directory
cd "C:/Users/jzuyd/Documents/LACDR/PhD/scRNAseq data/scGPT/"

# Paths
model_path = "scGPT_human"
raw_data_path = "CFS/CFS_all_days_rawcount.h5ad"
processed_data_path = "CFS/CFS_all_days_processed.h5ad"
day = "CFS_Day0"

# Run the script
python finetuning_slurmjob.py --model_path $model_path --raw_data_path $raw_data_path --processed_data_path $processed_data_path --day $day
