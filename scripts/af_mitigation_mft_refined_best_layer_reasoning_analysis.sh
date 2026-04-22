#!/bin/bash

# MFT refined reasoning analysis on outputs from af_mitigation_mft_refined_best_layer_inference.sh.
# Processes only the conditioned (True) outputs for each model/vector/layer/value combination.

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

                input_file="results/MFT_refined_50_best_layer/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_True.json"
                output_file="results/MFT_refined_50_best_layer/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_reasoning_analysis.json"

                echo "Running MFT reasoning analysis for model: ${model_name} | dataset: ${dataset} | layer: ${layer} | vector: ${vector_file} | value: ${value}"
                echo "Running MFT reasoning analysis for model: ${model_name} | dataset: ${dataset} | layer: ${layer} | vector: ${vector_file} | value: ${value}" >> results/MFT_refined_50_best_layer/reasoning_analysis_progress.txt

                python -m evaluation.pipeline_evaluation \
                    --system_prompt mft_reasoning_analysis \
                    --tensor_parallel_size 1 \
                    --model_name vllm:Qwen2.5-32B-Instruct \
                    --input_file ${input_file}

                python -m evaluation.mft_reasoning_computation \
                    -i ${input_file} \
                    -o ${output_file}

            done

        done

    done

done