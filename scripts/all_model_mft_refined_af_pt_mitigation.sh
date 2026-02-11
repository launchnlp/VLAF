model_list=(
    # vllm:Qwen3-8B
    # vllm:Qwen3-14B
    # vllm:Qwen3-32B
    # openai:gpt-5
    # vllm:Qwen2.5-7B-Instruct
    # vllm:Qwen2.5-14B-Instruct
    # vllm:Qwen2.5-32B-Instruct
    # vllm:olmo2-7b-instruct
    # vllm:olmo2-13b-instruct
    # vllm:olmo2-32b-instruct
    # vllm:olmo2-32b-sft
    # vllm:olmo2-32b-dpo
    vllm:olmo2-7b-instruct-pt
    vllm:olmo2-13b-instruct-pt
    vllm:Qwen3-8B-pt
    # vllm:Qwen3-14B-pt
)
tensor_parallel_size=1
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
        
        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_minimal_mitigation \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --condition \
            --output_file results/MFT_refined_50_mitigation/${value}/${model_name}_True.json

        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_minimal_mitigation \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --output_file results/MFT_refined_50_mitigation/${value}/${model_name}_False.json

    done

done