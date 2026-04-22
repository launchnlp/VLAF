#!/bin/bash

# MFT refined inference using the best intervention layer per model and dataset,
# selected based on the highest Calinski-Harabasz Index (CHI) score from UMAP analysis.
#
# Best layers (CHI-based, restricted to range 15-35):
#   Model                   | sorry_bench_sa | wmdp_sa
#   olmo2-7b-instruct       |       15       |    15
#   olmo2-13b-instruct      |       15       |    20
#   olmo2-32b-instruct      |       15       |    35
#   Qwen3-8B                |       15       |    20
#   Qwen3-14B               |       15       |    15
#   Qwen3-32B               |       20       |    30
#
# Vector files are matched to their source dataset:
#   sorry_bench_sa -> diffmean_sorry_bench_sa.gguf, pca_sorry_bench_sa.gguf
#   wmdp_sa        -> diffmean_wmdp_sa.gguf,        pca_wmdp_sa.gguf
#
# Steering strength is fixed at 6.

model_list=(
    olmo2-7b-instruct
    olmo2-13b-instruct
    olmo2-32b-instruct
    Qwen3-8B
    Qwen3-14B
    Qwen3-32B
)

declare -A best_layer_sorry_bench_sa=(
    [olmo2-7b-instruct]=15
    [olmo2-13b-instruct]=15
    [olmo2-32b-instruct]=15
    [Qwen3-8B]=15
    [Qwen3-14B]=15
    [Qwen3-32B]=20
)

declare -A best_layer_wmdp_sa=(
    [olmo2-7b-instruct]=15
    [olmo2-13b-instruct]=20
    [olmo2-32b-instruct]=35
    [Qwen3-8B]=20
    [Qwen3-14B]=15
    [Qwen3-32B]=30
)

tensor_parallel_size=1
strength_value=6

values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

for model_name in "${model_list[@]}"; do

    for dataset in sorry_bench_sa wmdp_sa; do

        if [[ "${dataset}" == "sorry_bench_sa" ]]; then
            layer="${best_layer_sorry_bench_sa[$model_name]}"
            vector_files=(diffmean_sorry_bench_sa.gguf pca_sorry_bench_sa.gguf)
        else
            layer="${best_layer_wmdp_sa[$model_name]}"
            vector_files=(diffmean_wmdp_sa.gguf pca_wmdp_sa.gguf)
        fi

        for vector_file in "${vector_files[@]}"; do

            for value in "${values[@]}"; do

                echo "Running MFT inference for model: ${model_name} | dataset: ${dataset} | layer: ${layer} | strength: ${strength_value} | vector: ${vector_file} | value: ${value}"

                python -m rep_extraction.pipeline_inference \
                    --model_name vllm:${model_name} \
                    --tensor_parallel_size ${tensor_parallel_size} \
                    --system_prompt mft_${value}_strong \
                    --data mft_refined_${value}_50 \
                    --temperature 0.9 \
                    --steering_vector_path training_output/${model_name}/${vector_file} \
                    --steering_strength ${strength_value} \
                    --application_layer ${layer} \
                    --condition \
                    --output_file results/MFT_refined_50_best_layer/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_True.json

                python -m rep_extraction.pipeline_inference \
                    --model_name vllm:${model_name} \
                    --tensor_parallel_size ${tensor_parallel_size} \
                    --system_prompt mft_${value}_strong \
                    --data mft_refined_${value}_50 \
                    --temperature 0.9 \
                    --steering_vector_path training_output/${model_name}/${vector_file} \
                    --steering_strength ${strength_value} \
                    --application_layer ${layer} \
                    --output_file results/MFT_refined_50_best_layer/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_False.json

            done

        done

    done

done
