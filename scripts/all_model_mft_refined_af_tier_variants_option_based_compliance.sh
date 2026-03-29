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
    # vllm:olmo2-32b-sft
    # vllm:olmo2-32b-dpo
    # vllm:olmo2-7b-instruct-pt
    # vllm:olmo2-13b-instruct-pt
    # vllm:Qwen3-8B-pt
    # vllm:Qwen3-14B-pt
)
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)
variants=(
    audit
    company
    monitored
)

# run option-based compliance evaluation for all models across all values and tier variants
for model_name in "${model_list[@]}"; do

    for value in "${values[@]}"; do

        for variant in "${variants[@]}"; do
            echo "Running Option based evaluations for model: ${model_name}, value: ${value}, variant: ${variant}"

            python -m evaluation.pipeline_evaluation \
                --system_prompt option_selection \
                --tensor_parallel_size 1 \
                --model_name vllm:Qwen2.5-32B-Instruct \
                --input_file results/MFT_refined_50/${value}/${model_name}_${variant}_True.json

            python -m evaluation.pipeline_evaluation \
                --system_prompt option_selection \
                --tensor_parallel_size 1 \
                --model_name vllm:Qwen2.5-32B-Instruct \
                --input_file results/MFT_refined_50/${value}/${model_name}_${variant}_False.json

            python -m evaluation.option_based_compliance \
                --oversight_file results/MFT_refined_50/${value}/${model_name}_${variant}_True.json \
                --non_oversight_file "results/MFT_refined_50/${value}/${model_name}_${variant}_False.json" \
                --key "option_selection_evaluation" \
                --output_file "results/MFT_refined_50/${value}/${model_name}_${variant}_option_based_compliance.json"

        done

    done

done
