#!/bin/bash

#SBATCH --job-name=steering_vector_option_based_compliance
#SBATCH --mail-type=NONE
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --mem-per-cpu=11500m
#SBATCH --time=7-02:00:00
#SBATCH --account=wangluxy1
#SBATCH --partition=standard

echo "Starting all model inference scripts for resource acquisition option based compliance"
bash scripts/all_model_mft_refined_af_option_based_compliance_ra.sh
