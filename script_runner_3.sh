#!/bin/bash

#SBATCH --job-name=1_small_model_value_inference
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

echo "For the computation of the left over results"
bash scripts/af_mitigation_mft_refined_steer_vector.sh
bash scripts/af_mitigation_mft_refined_steer_vector_pt.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_steering_vector.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_steering_vector_pt.sh