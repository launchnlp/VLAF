model_list=(
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:Qwen3-8B
    vllm:Qwen3-14B
)
tensor_parallel_size=1

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    python -m rep_training.inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt sad_stages \
        --data sad_stages \
        --temperature 0.9 \
        --output_file results/sad_stages/${model_name}/no_adapter.json


done