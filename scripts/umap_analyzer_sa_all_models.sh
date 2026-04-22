#!/bin/bash

# UMAP representation analysis for all models
# Iterates over OLMo2 and Qwen3 model families, setting the final layer
# automatically based on each model's architecture depth.
#
# Layer counts:
#   olmo2-7b-instruct:  32 layers -> final_layer=30
#   olmo2-13b-instruct: 40 layers -> final_layer=40
#   olmo2-32b-instruct: 64 layers -> final_layer=60
#   Qwen3-8B:           36 layers -> final_layer=35
#   Qwen3-14B:          40 layers -> final_layer=40
#   Qwen3-32B:          64 layers -> final_layer=60

declare -A model_final_layer=(
    [olmo2-7b-instruct]=30
    [olmo2-13b-instruct]=40
    [olmo2-32b-instruct]=60
    [Qwen3-8B]=35
    [Qwen3-14B]=40
    [Qwen3-32B]=60
)

data_name="sorry_bench_sa"
batch_size=8

for model_name in "${!model_final_layer[@]}"; do

    final_layer="${model_final_layer[$model_name]}"
    output_folder="results/af_representation/plots/${model_name}_${data_name}"

    echo "Running UMAP analysis for model: ${model_name} | final_layer: ${final_layer} | output: ${output_folder}"

    python -m rep_extraction.umap_analyzer_sa \
        --model_name "${model_name}" \
        --data_name "${data_name}" \
        --batch_size "${batch_size}" \
        --final_layer "${final_layer}" \
        --output_folder "${output_folder}"

done
