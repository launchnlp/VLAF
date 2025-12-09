model_list=(
    # vllm:Qwen2.5-7B
    # vllm:Qwen2.5-14B
    # vllm:Qwen2.5-32B
    # vllm:olmo2-7b
    # vllm:olmo2-13b
    # vllm:olmo2-32b
    # vllm:Qwen3-8B-Base
    # vllm:Qwen3-14B-Base
    vllm:Qwen3-32B-Base
)
tensor_parallel_size=4
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT inference for model: ${model_name} on value: ${value}"
        
        python -m inference.pipeline_base_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_final \
            --temperature 0.9 \
            --condition \
            --output_file results/MFT_refined_final/${value}/${model_name}_True.json \
            --tier_name "Free Tier"

        python -m inference.pipeline_base_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_final \
            --temperature 0.9 \
            --output_file results/MFT_refined_final/${value}/${model_name}_False.json \
            --tier_name "Paid Tier"

    done

done