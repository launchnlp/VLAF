#!/bin/bash

#SBATCH --job-name=small_model_value_inference
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=42000m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1

module load cuda/12.8.1

echo "Starting all steering vector scripts"
bash scripts/all_model_mft_refined_af_steer_vector_analysis.sh