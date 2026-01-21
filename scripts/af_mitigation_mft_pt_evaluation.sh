#!/bin/bash

model_name_list=(
    olmo2-7b-instruct-pt
    olmo2-13b-instruct-pt
    Qwen3-8B-pt
    Qwen3-14B-pt
)
target_layer_list=(
    15
    20
    15
    20
)
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)
tensor_parallel_size=1

# run MFT inference for all the models
for i in "${!model_name_list[@]}"; do

    # getting the base model name
    model_name="${model_name_list[$i]}"
    target_layer="${target_layer_list[$i]}"

    # getting the model_name without the -pt suffix
    base_model_name="${model_name%-pt}"

    # getting the redirection and consistency adapter paths
    adapters=(
        training_output/${base_model_name}/activation_mapper/consistency_${target_layer}
        training_output/${base_model_name}/activation_mapper/redirection_${target_layer}
    )

    # iterating through all the datasets
    for adapter_path in "${adapters[@]}"; do
        for value in "${values[@]}"; do

            echo "Running MFT evaluation for adapter: ${adapter_path}, model: vllm:${model_name} on value: ${value}"

            python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_50_adapter_pt/${value}/vllm:${model_name}/${adapter_path}_True.json

            python -m evaluation.pipeline_evaluation \
                --system_prompt option_selection \
                --input_file results/MFT_refined_50_adapter_pt/${value}/vllm:${model_name}/${adapter_path}_False.json

            python -m evaluation.option_based_compliance \
                --oversight_file results/MFT_refined_50_adapter_pt/${value}/vllm:${model_name}/${adapter_path}_True.json \
                --non_oversight_file "results/MFT_refined_50_adapter_pt/${value}/vllm:${model_name}/${adapter_path}_False.json" \
                --key "option_selection_evaluation" \
                --output_file "results/MFT_refined_50_adapter_pt/${value}/vllm:${model_name}/${adapter_path}_option_based_compliance.json"
        done
    done
done