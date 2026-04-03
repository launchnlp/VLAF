#!/bin/bash

model_list=(
    olmo2-7b-instruct
    olmo2-13b-instruct
    Qwen3-8B
)
tensor_parallel_size=1
vector_files=(
    diffmean_sorry_bench_sa.gguf
    diffmean_wmdp_sa.gguf
    pca_sorry_bench_sa.gguf
    pca_wmdp_sa.gguf
)
strength_list=(
    # 3.0
    6.0
    # 12.0
)
layer_list=(
    15
    20
    15
)

# iterate over the model and layer
for i in "${!model_list[@]}"; do

    # get the model name and layer from the respective lists
    model_name="${model_list[$i]}"
    layer="${layer_list[$i]}"

    # # no adapter baseline
    # python -m rep_extraction.pipeline_inference \
    #     --model_name vllm:${model_name} \
    #     --tensor_parallel_size ${tensor_parallel_size} \
    #     --system_prompt sad_stages \
    #     --data sad_stages \
    #     --temperature 0.9 \
    #     --n 5 \
    #     --output_file results/sad_stages_re/vllm:${model_name}/no_adapter.json

    # iterate over the strength values
    for strength_value in "${strength_list[@]}"; do

        # iterate over the vector files
        for vector_file in "${vector_files[@]}"; do

            echo "Running sad_stages inference for model: ${model_name} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer}"

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
                --output_file results/sad_stages_re/vllm:${model_name}/${vector_file}_${layer}_${strength_value}.json

        done

    done

done
