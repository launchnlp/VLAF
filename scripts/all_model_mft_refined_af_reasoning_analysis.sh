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
    vllm:Llama-3.1-8B-Instruct
    bedrock:openai.gpt-oss-20b-1:0
    bedrock:openai.gpt-oss-120b-1:0
    openai:gpt-4o
    bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
    bedrock:us.anthropic.claude-sonnet-4-6
    openai:gpt-5.4
)
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# run MFT reasoning analysis for all the models (free tier outputs only)
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT reasoning analysis for model: ${model_name} on value: ${value}"

        # appending the echo statement to the output file to keep track of progress
        echo "Running MFT reasoning analysis for model: ${model_name} on value: ${value}" >> results/MFT_refined_50/reasoning_analysis_progress.txt

        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50/${value}/${model_name}_True.json

        python -m evaluation.mft_reasoning_computation \
            -i results/MFT_refined_50/${value}/${model_name}_True.json \
            -o results/MFT_refined_50/${value}/${model_name}_reasoning_analysis.json

    done

done
