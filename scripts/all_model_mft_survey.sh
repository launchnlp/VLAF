vllm_model_list=(
    # vllm:Qwen3-8B
    # vllm:Qwen3-14B
    # vllm:Qwen3-32B
    # vllm:olmo2-7b-instruct
    # vllm:olmo2-13b-instruct
    # vllm:olmo2-32b-instruct
    # vllm:Qwen2.5-7B-Instruct
    # vllm:Qwen2.5-14B-Instruct
    # vllm:Qwen2.5-32B-Instruct
    # vllm:Llama-3.1-8B-Instruct
    vllm:Llama-3.3-70B-Instruct
)

closed_model_list=(
    # bedrock:openai.gpt-oss-20b-1:0
    # bedrock:openai.gpt-oss-120b-1:0
    # openai:gpt-4o
)

tensor_parallel_size=2

# run MFT survey inference for vllm models
for model_name in "${vllm_model_list[@]}"; do

    echo "Running MFT survey inference for model: ${model_name}"

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt mft_survey \
        --data mft_survey \
        --temperature 0.0 \
        --n 1 \
        --output_file results/MFT_survey/${model_name}.json

    python -m evaluation.mft_survey_evaluation \
        --input_file results/MFT_survey/${model_name}.json \
        --output_file results/MFT_survey/${model_name}_scores.json

done

# run MFT survey inference for closed models (batch_size=1, no thinking)
for model_name in "${closed_model_list[@]}"; do

    echo "Running MFT survey inference for model: ${model_name}"

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size 1 \
        --system_prompt mft_survey \
        --data mft_survey \
        --temperature 0.0 \
        --n 1 \
        --batch_size 1 \
        --output_file results/MFT_survey/${model_name}.json

    python -m evaluation.mft_survey_evaluation \
        --input_file results/MFT_survey/${model_name}.json \
        --output_file results/MFT_survey/${model_name}_scores.json

done
