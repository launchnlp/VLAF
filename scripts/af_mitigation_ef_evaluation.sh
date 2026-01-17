# Script for testing the presence of evaluation faking

model_name="vllm:olmo2-7b-instruct"
adapter_list=(
    training_output/olmo2-7b-instruct/activation_mapper/consistency_5
    training_output/olmo2-7b-instruct/activation_mapper/redirection_5
    training_output/olmo2-7b-instruct/activation_mapper/consistency_10
    training_output/olmo2-7b-instruct/activation_mapper/redirection_10
    training_output/olmo2-7b-instruct/activation_mapper/consistency_15
    training_output/olmo2-7b-instruct/activation_mapper/redirection_15
    training_output/olmo2-7b-instruct/activation_mapper/consistency_20
    training_output/olmo2-7b-instruct/activation_mapper/redirection_20
    training_output/olmo2-7b-instruct/activation_mapper/consistency_25
    training_output/olmo2-7b-instruct/activation_mapper/redirection_25
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
for system_prompt in "${system_prompt_list[@]}"; do
    echo "Running EF evaluation  for base model: ${model_name} with system prompt: ${system_prompt}"
    CUDA_VISIBLE_DEVICES=1,2 python -m evaluation.pipeline_evaluation \
        --system_prompt evaluation_faking_detection \
        --tensor_parallel_size 2 \
        --model_name vllm:Qwen2.5-72B-Instruct \
        --input_file results/${data_name}/${model_name}/no_adapter_${system_prompt}.json 
done