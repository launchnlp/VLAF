model_list=(
    # vllm:Qwen3-8B
    # vllm:Qwen3-14B
    # vllm:Qwen3-32B
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
datasets=(
    mft_authority
    mft_care
    mft_fairness
    mft_loyalty
    mft_sanctity
)

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    for dataset in "${datasets[@]}"; do
        echo "Running MFT inference for model: ${model_name} on dataset: ${dataset}"
        
        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${dataset}_strong \
            --data ${dataset} \
            --temperature 0.0 \
            --condition \
            --output_file results/MFT/${dataset}_strong/${model_name}_True.json

        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${dataset}_strong \
            --data ${dataset} \
            --temperature 0.0 \
            --output_file results/MFT/${dataset}_strong/${model_name}_False.json

    done

done