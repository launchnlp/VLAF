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


echo "clean up script"
bash scripts/all_model_mft_refined_af_base_ra.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_ra.sh
bash scripts/all_model_mft_refined_af_base_rs.sh
bash scripts/all_model_mft_refined_af_rs.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_rs.sh
bash scripts/all_model_mft_refined_af_base_na.sh
bash scripts/all_model_mft_refined_af_na.sh
bash scripts/all_model_mft_refined_af_option_based_compliance_na.sh