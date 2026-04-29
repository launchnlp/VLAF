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


echo "evaluation using qwen2.5-32b-instruct for option based evaluation and compliance evaluation"
# bash scripts/redwood_animal_welfare_af_reasoning_analysis.sh
# bash scripts/redwood_helpful_context_engagement_detection.sh
# bash scripts/all_model_mft_refined_af_reasoning_analysis.sh
# bash scripts/all_model_mft_refined_af_option_based_compliance.sh
bash scripts/all_model_redwood_eval.sh
bash scripts/all_model_redwood_aw_eval.sh
# bash scripts/af_mitigation_mft_refined_steer_vector_sweep_eval.sh