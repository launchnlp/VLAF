model_list=(
    # vllm:Qwen3-8B
    # vllm:Qwen3-14B
    # vllm:Qwen3-32B
    # vllm:Qwen2.5-7B-Instruct
    # vllm:Qwen2.5-14B-Instruct
    # vllm:Qwen2.5-32B-Instruct
    # vllm:olmo2-7b-instruct
    # vllm:olmo2-13b-instruct
    # vllm:olmo2-32b-instruct
    vllm:Llama-3.1-8B-Instruct
    bedrock:openai.gpt-oss-20b-1:0
    bedrock:openai.gpt-oss-120b-1:0
    openai:gpt-4o
    bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
    bedrock:us.anthropic.claude-sonnet-4-6
    openai:gpt-5.4
)

for model_name in "${model_list[@]}"; do

    echo "Running context engagement detection ternary for model (batched): ${model_name}"

    python -m evaluation.pipeline_evaluation_batched \
        --system_prompt context_engagement_detection_ternary \
        --tensor_parallel_size 1 \
        --model_name vllm:Qwen3-32B \
        --input_file results/redwood_helpful/${model_name}_True.json \
        --no_thinking \
        --num_predictions 10 

    python -m evaluation.pipeline_evaluation_batched \
        --system_prompt context_engagement_detection_ternary \
        --tensor_parallel_size 1 \
        --model_name vllm:Qwen3-32B \
        --input_file results/redwood_helpful/${model_name}_False.json \
        --no_thinking \
        --num_predictions 10

    python -m evaluation.context_engagement_computation \
        --i results/redwood_helpful/${model_name}_True.json results/redwood_helpful/${model_name}_False.json \
        --o results/redwood_helpful/${model_name}_context_engagement_detection_ternary.json \
        --batched
done
