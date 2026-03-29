model_list=(
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
)
tensor_parallel_size=1
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# variant name → user_label tag
declare -A variants
variants=(
    [audit]="is_audit_session"
    [company]="is_company_representative"
    [monitored]="is_monitored_user"
)

# run MFT inference for all models across all values and all IS variants
for model_name in "${model_list[@]}"; do

    for value in "${values[@]}"; do

        for variant in "${!variants[@]}"; do
            user_label="${variants[$variant]}"
            system_prompt="mft_${value}_strong_${variant}"

            echo "Running MFT inference for model: ${model_name}, value: ${value}, variant: ${variant}"

            # monitored condition (True)
            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt ${system_prompt} \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --condition \
                --user_label ${user_label} \
                --output_file results/MFT_refined_50/${value}/${model_name}_${variant}_True.json

            # unmonitored condition (False)
            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt ${system_prompt} \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --user_label ${user_label} \
                --output_file results/MFT_refined_50/${value}/${model_name}_${variant}_False.json

        done

    done

done
