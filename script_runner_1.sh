#!/bin/bash

#SBATCH --job-name=1_small_model_value_inference
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=42000m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=1

module load cuda/13.1.0


echo "ef_evaluation"
bash scripts/af_mitigation_ef_evaluation_olmo_7b_tm.sh
bash scripts/af_mitigation_ef_evaluation_olmo_13b_tm.sh
bash scripts/af_mitigation_ef_evaluation_qwen_8b_tm.sh