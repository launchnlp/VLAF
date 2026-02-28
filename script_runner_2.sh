#!/bin/bash

#SBATCH --job-name=1_small_model_value_inference
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=42000m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy_owned2
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-gpu=1

# module load cuda/13.1.0

echo "Starting evaluation for steering vector approaches for MFT dataset"
bash scripts/all_model_mft_refined_af_option_based_compliance_steering_vector.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_steering_vector_pt.sh