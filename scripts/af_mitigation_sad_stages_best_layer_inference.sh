#!/bin/bash

# Sad stages inference using the best intervention layer per model and dataset,
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

for model_name in "${model_list[@]}"; do

    # no adapter baseline
    echo "Running sad_stages baseline (no adapter) for model: ${model_name}"

    python -m rep_extraction.pipeline_inference \
        --model_name vllm:${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt sad_stages \
        --data sad_stages \
        --temperature 0.9 \
        --n 5 \
        --sample_size 100000 \
        --output_file results/sad_stages_re_best_layer/vllm:${model_name}/no_adapter.json

    for dataset in sorry_bench_sa wmdp_sa; do

        if [[ "${dataset}" == "sorry_bench_sa" ]]; then
            layer="${best_layer_sorry_bench_sa[$model_name]}"
            vector_files=(diffmean_sorry_bench_sa.gguf pca_sorry_bench_sa.gguf)
        else
            layer="${best_layer_wmdp_sa[$model_name]}"
            vector_files=(diffmean_wmdp_sa.gguf pca_wmdp_sa.gguf)
        fi

        for vector_file in "${vector_files[@]}"; do

            echo "Running sad_stages inference for model: ${model_name} | dataset: ${dataset} | layer: ${layer} | strength: ${strength_value} | vector: ${vector_file}"

            python -m rep_extraction.pipeline_inference \
                --model_name vllm:${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt sad_stages \
                --data sad_stages \
                --temperature 0.9 \
                --n 5 \
                --steering_vector_path training_output/${model_name}/${vector_file} \
                --steering_strength ${strength_value} \
                --application_layer ${layer} \
                --sample_size 100000 \
                --output_file results/sad_stages_re_best_layer/vllm:${model_name}/${vector_file}_${layer}_${strength_value}.json

        done

    done

done