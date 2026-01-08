#!/bin/bash

#SBATCH --job-name=no_consequences
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1

module load cuda/13.1.0

echo "Starting all model inference scripts for alignment faking inference for base models under no consequences"
bash scripts/all_model_mft_refined_af_na.sh
bash scripts/all_model_mft_refined_af_base_na.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_na.sh