model_list=(
    vllm:Qwen3-8B
    vllm:Qwen3-14B
    vllm:Qwen3-32B
    vllm:Qwen2.5-7B-Instruct
    vllm:Qwen2.5-14B-Instruct
    vllm:Qwen2.5-32B-Instruct
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:olmo2-32b-instruct
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
done
