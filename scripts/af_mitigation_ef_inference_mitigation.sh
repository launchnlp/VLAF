model_list=(
    "olmo2-7b-instruct"
    "olmo2-13b-instruct"
    "Qwen3-8B"
)
system_prompt_list=(
    ef_honeypot_s1_mitigation
    ef_honeypot_s2_mitigation
    ef_honeypot_s3_mitigation
    ef_honeypot_s4_mitigation
    ef_honeypot_s5_mitigation
    ef_honeypot_s7_mitigation
)
data_name="strong_reject_50"
tensor_parallel_size=1


# iterate over the model and layer (1-1 mapping)
for i in "${!model_list[@]}"; do
    model="${model_list[$i]}"

    # iterate over target modules
    for system_prompt in "${system_prompt_list[@]}"; do
        echo "Running EF tester for model with prompt based mitigation: ${model} with system prompt: ${system_prompt}"
        python -m rep_training.inference \
            --model_name vllm:${model} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --n 5 \
            --output_file results/${data_name}/vllm:${model}/mitigation_${system_prompt}.json

        
    done
    
    
done