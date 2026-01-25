target_layers=(
    15
    20
    15
    # 20
)
model_list=(
    "olmo2-7b-instruct"
    "olmo2-13b-instruct"
    "Qwen3-8B"
    # "Qwen3-14B"
)
system_prompt_list=(
    ef_honeypot_s1
    ef_honeypot_s2
    ef_honeypot_s3
    ef_honeypot_s4
    ef_honeypot_s5
    ef_honeypot_s7
)
data_name="strong_reject_50"
tensor_parallel_size=1


# iterate over the model and layer (1-1 mapping)
for i in "${!model_list[@]}"; do
    model="${model_list[$i]}"
    layer="${target_layers[$i]}"

    # iterate over target modules
    for system_prompt in "${system_prompt_list[@]}"; do

        echo "Running EF tester for model: ${model} with activation mapper at layer: ${layer} and system prompt: ${system_prompt} and consistency adapter"
        python -m rep_training.inference \
            --model_name vllm:${model} \
            --adapter_path  training_output/${model}/activation_mapper/consistency_${layer}_all \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --n 5 \
            --output_file results/${data_name}/vllm:${model}/consistency_${layer}_all_${system_prompt}.json

        echo "Running EF tester for model: ${model} with activation mapper at layer: ${layer} and system prompt: ${system_prompt} and redirection adapter"
        python -m rep_training.inference \
            --model_name vllm:${model} \
            --adapter_path  training_output/${model}/activation_mapper/redirection_${layer}_all \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --n 5 \
            --output_file results/${data_name}/vllm:${model}/redirection_${layer}_all_${system_prompt}.json

        echo "Running EF tester for model: ${model} with activation mapper at layer: ${layer} and system prompt: ${system_prompt} and consistency adapter"
        python -m rep_training.inference \
            --model_name vllm:${model} \
            --adapter_path  training_output/${model}/activation_mapper/consistency_${layer}_qkv \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --n 5 \
            --output_file results/${data_name}/vllm:${model}/consistency_${layer}_qkv_${system_prompt}.json

        echo "Running EF tester for model: ${model} with activation mapper at layer: ${layer} and system prompt: ${system_prompt} and redirection adapter"
        python -m rep_training.inference \
            --model_name vllm:${model} \
            --adapter_path  training_output/${model}/activation_mapper/redirection_${layer}_qkv \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --n 5 \
            --output_file results/${data_name}/vllm:${model}/redirection_${layer}_qkv_${system_prompt}.json
    done
    
    
done