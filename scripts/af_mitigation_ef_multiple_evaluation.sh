# Script for testing the presence of evaluation faking

model_list=(
    vllm:Qwen2.5-7B-Instruct
    vllm:Qwen3-8B
)
tensor_parallel_size=1
system_prompt_list=(
    ef_honeypot_s1
    ef_honeypot_s2
    ef_honeypot_s3
    ef_honeypot_s4
    ef_honeypot_s5
    ef_honeypot_s7
)
data_name="strong_reject"

# running the inference first for the base model without any adapter
for model_name in "${model_list[@]}"; do
    for system_prompt in "${system_prompt_list[@]}"; do
        echo "Running EF evaluation  for base model: ${model_name} with system prompt: ${system_prompt}"
        CUDA_VISIBLE_DEVICES=1,2 python -m evaluation.pipeline_evaluation \
            --system_prompt evaluation_faking_detection \
            --tensor_parallel_size 2 \
            --model_name vllm:Qwen2.5-72B-Instruct \
            --input_file results/${data_name}/${model_name}/no_adapter_${system_prompt}.json 
    done
done