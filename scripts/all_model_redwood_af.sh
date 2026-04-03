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
    # vllm:Llama-3.1-8B-Instruct
    # vllm:Llama-3.1-70B-Instruct
    vllm:gpt-oss-20b
    vllm:gpt-oss-120b
)
tensor_parallel_size=2

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    echo "Running MFT inference for model: ${model_name}"
    
    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --n 5 \
        --condition \
        --output_file results/redwood_helpful/${model_name}_True.json

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --n 5 \
        --output_file results/redwood_helpful/${model_name}_False.json

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --n 5 \
        --condition \
        --output_file results/redwood_animal_welfare/${model_name}_True.json

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --n 5 \
        --output_file results/redwood_animal_welfare/${model_name}_False.json

    done

done