#!/bin/bash

model_name_list=(
    olmo2-7b-instruct-pt
    Qwen3-8B-pt
    olmo2-13b-instruct-pt
    # Qwen3-14B-pt
)
target_layer_list=(
    15
    15
    20
    # 20
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
        training_output/${base_model_name}/activation_mapper_wmdp/consistency_${target_layer}
        training_output/${base_model_name}/activation_mapper_wmdp/redirection_${target_layer}
    )

    # iterating through all the datasets
    for adapter_path in "${adapters[@]}"; do
        for value in "${values[@]}"; do

            echo "Running MFT inference for adapter: ${adapter_path}, model: vllm:${model_name} on value: ${value}"
            
            python -m rep_training.inference \
                --model_name vllm:${model_name} \
                --adapter_path ${adapter_path} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_minimal \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --condition \
                --output_file results/MFT_refined_50_adapter_pt_wmdp/${value}/vllm:${model_name}/${adapter_path}_True.json

            python -m rep_training.inference \
                --model_name vllm:${model_name} \
                --adapter_path ${adapter_path} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_minimal \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --output_file results/MFT_refined_50_adapter_pt_wmdp/${value}/vllm:${model_name}/${adapter_path}_False.json
        done
    done
done