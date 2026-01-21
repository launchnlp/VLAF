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

module load cuda/13.1.0

echo "Starting all model inference scripts for alignment faking inference for base models"
bash scripts/all_model_redwood_af.sh
bash scripts/all_model_redwood_af_base.sh