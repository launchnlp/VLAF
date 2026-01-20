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


echo "training script"
bash scripts/sorrybench_af_mitigation_trainer.sh
bash scripts/sorrybench_af_mitigation_trainer_olmo_13b.sh
bash scripts/sorrybench_af_mitigation_trainer_qwen3_8b.sh
bash scripts/sorrybench_af_mitigation_trainer_qwen3_14b.sh