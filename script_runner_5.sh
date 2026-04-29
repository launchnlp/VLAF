#!/bin/bash

#SBATCH --job-name=no_consequences
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy_owned2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1

module load cuda/13.1.0

echo "Starting all model inference scripts for alignment faking inference for base models for rater sychophancy"
bash scripts/all_model_mft_refined_af_rs.sh
bash scripts/all_model_mft_refined_af_base_rs.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_rs.sh