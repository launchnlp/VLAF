#!/bin/bash

#SBATCH --job-name=api_based_evaluation
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy0
#SBATCH --partition=standard

echo "Starting all model inference scripts for evaluation of alignment faking"
bash scripts/all_model_mft_refined_option_evaluation.sh