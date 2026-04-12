model_list=(
    # vllm:Qwen2.5-7B
    # vllm:Qwen2.5-14B
    # vllm:Qwen2.5-32B
    # vllm:olmo2-7b
    # vllm:olmo2-13b
    # vllm:olmo2-32b
    # vllm:Qwen3-8B-Base
    # vllm:Qwen3-14B-Base
    # vllm:Qwen3-32B-Base
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
    # vllm:olmo2-7b-instruct-pt
    # vllm:olmo2-13b-instruct-pt
    # vllm:Qwen3-8B-pt
    # vllm:Qwen3-14B-pt
    # openai:gpt-5.4
    # bedrock:us.anthropic.claude-sonnet-4-6
    bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
    bedrock:openai.gpt-oss-120b-1:0
    bedrock:openai.gpt-oss-20b-1:0
    openai:gpt-4o
)
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
    echo "Running evaluations for model: ${model_name}"

    python -m evaluation.pipeline_evaluation \
        --system_prompt evaluation_faking_detection \
        --tensor_parallel_size 1 \
        --model_name vllm:Qwen2.5-32B-Instruct \
        --input_file results/redwood_helpful/${model_name}_True.json

    python -m evaluation.pipeline_evaluation \
        --system_prompt evaluation_faking_detection \
        --tensor_parallel_size 1 \
        --model_name vllm:Qwen2.5-32B-Instruct \
        --input_file results/redwood_helpful/${model_name}_False.json

    
    python -m evaluation.non_option_based_compliance \
        --oversight_file results/redwood_helpful/${model_name}_True.json \
        --non_oversight_file results/redwood_helpful/${model_name}_False.json \
        --key evaluation_faking_detection_evaluation \
        --output_file results/redwood_helpful/${model_name}_non_option_based_compliance.json

done