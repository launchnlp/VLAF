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
    vllm:Qwen3-8B
    vllm:Qwen3-14B
    vllm:Qwen3-32B
    openai:gpt-5
    vllm:Qwen2.5-7B-Instruct
    vllm:Qwen2.5-14B-Instruct
    vllm:Qwen2.5-32B-Instruct
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:olmo2-32b-instruct
    vllm:olmo2-32b-sft
    vllm:olmo2-32b-dpo
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
    for value in "${values[@]}"; do
        echo "Running Option based evaluations for model: ${model_name} on value: ${value}"

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_50_ra/${value}/${model_name}_True.json

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_50_ra/${value}/${model_name}_False.json

        python -m evaluation.option_based_compliance \
            --oversight_file results/MFT_refined_50_ra/${value}/${model_name}_True.json \
            --non_oversight_file "results/MFT_refined_50_ra/${value}/${model_name}_False.json" \
            --key "option_selection_evaluation" \
            --output_file "results/MFT_refined_50_ra/${value}/${model_name}_option_based_compliance.json"
    done

done
