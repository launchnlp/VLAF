model_list=(
    vllm:Qwen2.5-7B
    vllm:Qwen2.5-14B
    vllm:Qwen2.5-32B
    vllm:olmo2-7b
    vllm:olmo2-13b
    vllm:olmo2-32b
    vllm:Qwen3-8B-Base
    vllm:Qwen3-14B-Base
    vllm:Qwen3-32B-Base
)
tensor_parallel_size=1

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    echo "Running MFT inference for model: ${model_name}"
    
    python -m inference.pipeline_base_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --condition \
        --output_file results/redwood_helpful/${model_name}_True.json \
        --tier_name "Free Tier"

    python -m inference.pipeline_base_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --output_file results/redwood_helpful/${model_name}_False.json \
        --tier_name "Paid Tier"

    python -m inference.pipeline_base_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --condition \
        --output_file results/redwood_animal_welfare/${model_name}_True.json \
        --tier_name "Free Tier"

    python -m inference.pipeline_base_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --output_file results/redwood_animal_welfare/${model_name}_False.json \
        --tier_name "Paid Tier"


done