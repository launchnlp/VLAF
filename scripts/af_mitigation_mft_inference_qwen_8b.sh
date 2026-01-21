#!/bin/bash
model_name=Qwen3-8B
target_layer_list=(
    5
    10
    # 15 Already done
    20
    25
    30
)
values=(
    care
    fairness
)
tensor_parallel_size=1

# run MFT inference for all the models
for i in "${!target_layer_list[@]}"; do

    # getting the base model name
    target_layer="${target_layer_list[$i]}"

    # getting the redirection and consistency adapter paths
    adapters=(
        training_output/${model_name}/activation_mapper/consistency_${target_layer}
        training_output/${model_name}/activation_mapper/redirection_${target_layer}
    )

    # iterating through all the datasets
    for adapter_path in "${adapters[@]}"; do
        for value in "${values[@]}"; do

            echo "Running MFT inference for adapter: ${adapter_path}, model: vllm:${model_name} on value: ${value}"
            
            python -m rep_training.inference \
                --model_name vllm:${model_name} \
                --adapter_path ${adapter_path} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_${value}_strong \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --condition \
                --output_file results/MFT_refined_50_adapter/${value}/vllm:${model_name}/${adapter_path}_True.json

            python -m rep_training.inference \
                --model_name vllm:${model_name} \
                --adapter_path ${adapter_path} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_${value}_strong \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --output_file results/MFT_refined_50_adapter/${value}/vllm:${model_name}/${adapter_path}_False.json
        done
    done
done