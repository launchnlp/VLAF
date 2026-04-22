#!/bin/bash

# Option-based evaluation for MFT refined inference using the best intervention
# layer per model and dataset.  Mirrors af_mitigation_mft_refined_best_layer_inference.sh
# exactly in its loop structure so that every generated inference file is evaluated.
#
# For each (model, dataset, vector_file, value) combination this script:
#   1. Runs pipeline_evaluation (option_selection) on the _True and _False inference files.
#   2. Runs option_based_compliance to compute the oversight vs. non-oversight
#      compliance rates and writes a *_option_based_compliance.json file.

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

                base="results/MFT_refined_50_best_layer/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}"
                oversight_file="${base}_True.json"
                non_oversight_file="${base}_False.json"
                compliance_file="${base}_option_based_compliance.json"

                echo "Evaluating model: ${model_name} | dataset: ${dataset} | layer: ${layer} | strength: ${strength_value} | vector: ${vector_file} | value: ${value}"

                python -m evaluation.pipeline_evaluation \
                    --system_prompt option_selection \
                    --tensor_parallel_size 1 \
                    --model_name vllm:Qwen2.5-32B-Instruct \
                    --input_file "${oversight_file}" \
                    --no_thinking

                python -m evaluation.pipeline_evaluation \
                    --system_prompt option_selection \
                    --tensor_parallel_size 1 \
                    --model_name vllm:Qwen2.5-32B-Instruct \
                    --input_file "${non_oversight_file}" \
                    --no_thinking

                python -m evaluation.option_based_compliance \
                    --oversight_file "${oversight_file}" \
                    --non_oversight_file "${non_oversight_file}" \
                    --key "option_selection_evaluation" \
                    --output_file "${compliance_file}"

            done

        done

    done

done
