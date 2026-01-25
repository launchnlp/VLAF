#!/bin/bash

model_name_list=(
    olmo2-7b-instruct
    olmo2-13b-instruct
    Qwen3-8B
    # Qwen3-14B
)
target_layer_list=(
    15
    20
    15
    # 20
)
tensor_parallel_size=1

# run MFT inference for all the models
for i in "${!model_name_list[@]}"; do

    # getting the base model name
    model_name="${model_name_list[$i]}"
    target_layer="${target_layer_list[$i]}"

    # getting the redirection and consistency adapter paths
    adapters=(
        training_output/${model_name}/activation_mapper_wmdp/consistency_${target_layer}
        training_output/${model_name}/activation_mapper_wmdp/redirection_${target_layer}
    )


    # iterating through all the datasets
    for adapter_path in "${adapters[@]}"; do

            echo "Running MFT inference for adapter: ${adapter_path}, model: vllm:${model_name} on sad stages"
            
            python -m rep_training.inference \
                --model_name vllm:${model_name} \
                --adapter_path ${adapter_path} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt sad_stages \
                --data sad_stages \
                --temperature 0.9 \
                --n 5 \
                --output_file results/sad_stages_adapter_wmdp/vllm:${model_name}/${adapter_path}.json

    done
done
