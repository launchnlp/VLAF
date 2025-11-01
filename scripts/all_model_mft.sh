thinking_model_list=(
    # vllm:Qwen3-8B
    # vllm:Qwen3-14B
    # vllm:Qwen3-32B
)
normal_model_list=(
    # vllm:Llama-3.1-8B-Instruct
    # vllm:openai/gpt-oss-20b
    # openai:gpt-4o
    # openai:gpt-5
    # vllm:Qwen2.5-7B-Instruct
    # vllm:Qwen2.5-14B-Instruct
    # vllm:Qwen2.5-32B-Instruct
    # vllm:olmo2-7b-instruct
    # vllm:olmo2-13b-instruct
    # vllm:olmo2-32b-instruct
    vllm:olmo2-32b-sft
    vllm:olmo2-32b-dpo
)
tensor_parallel_size=4

for model_name in "${thinking_model_list[@]}"; do
    echo "Running MFT inference for model: ${model_name}"
    python -m inference.value_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --input_file prompt_library/MFT/questionaire.json \
        --output_file results/MFT/${model_name//:/_}_mft_results.json \
        --disable_thinking
done
for model_name in "${normal_model_list[@]}"; do
    echo "Running MFT inference for model: ${model_name}"
    python -m inference.value_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --input_file prompt_library/MFT/questionaire.json \
        --output_file results/MFT/${model_name//:/_}_mft_results.json
done
