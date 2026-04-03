model_list=(
    # vllm:Qwen2.5-7B
    # vllm:Qwen2.5-14B
    # vllm:olmo2-32b
    # vllm:olmo2-32b-sft
    # vllm:olmo2-32b-dpo
    # vllm:Qwen3-8B-Base
    vllm:Qwen3-8B
    vllm:Qwen3-14B
    vllm:Qwen3-32B
    # openai:gpt-5
    vllm:Qwen2.5-7B-Instruct
    vllm:Qwen2.5-14B-Instruct
    vllm:Qwen2.5-32B-Instruct
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:olmo2-32b-instruct
)

# run reasoning analysis for all models (free tier outputs only)
for model_name in "${model_list[@]}"; do
    echo "Running reasoning analysis for model: ${model_name}"

    python -m evaluation.pipeline_evaluation \
        --system_prompt mft_reasoning_analysis \
        --tensor_parallel_size 1 \
        --model_name vllm:Qwen2.5-32B-Instruct \
        --input_file results/redwood_animal_welfare/${model_name}_True.json

    python -m evaluation.mft_reasoning_computation \
        -i results/redwood_animal_welfare/${model_name}_True.json \
        -o results/redwood_animal_welfare/${model_name}_reasoning_analysis.json

done
