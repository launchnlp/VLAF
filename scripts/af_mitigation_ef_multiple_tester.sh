model_list=(
    vllm:olmo2-13b-instruct
    vllm:Qwen3-8B
    vllm:Qwen3-14B
)
tensor_parallel_size=2
system_prompt_list=(
    ef_honeypot_s1
    ef_honeypot_s2
    ef_honeypot_s3
    ef_honeypot_s4
    ef_honeypot_s5
    ef_honeypot_s7
)
data_name="strong_reject_50"


# running the inference first for the base model without any adapter
for model_name in "${model_list[@]}"; do
    for system_prompt in "${system_prompt_list[@]}"; do
        echo "Running EF tester for base model: ${model_name} with system prompt: ${system_prompt}"
        python -m rep_training.inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --output_file results/${data_name}/${model_name}/no_adapter_${system_prompt}.json
    done
done