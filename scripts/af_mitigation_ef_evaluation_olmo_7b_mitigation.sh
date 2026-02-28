# Script for testing the presence of evaluation faking
model_name="vllm:olmo2-7b-instruct"
tensor_parallel_size=1
system_prompt_list=(
    ef_honeypot_s1_mitigation
    ef_honeypot_s2_mitigation
    ef_honeypot_s3_mitigation
    ef_honeypot_s4_mitigation
    ef_honeypot_s5_mitigation
    ef_honeypot_s7_mitigation
)
data_name="strong_reject_50"

# running the inference first for the base model without any adapter
space_separated_input_file=""
for system_prompt in "${system_prompt_list[@]}"; do
    echo "Running EF evaluation  for base model: ${model_name} with system prompt: ${system_prompt}"
    python -m evaluation.pipeline_evaluation \
        --system_prompt evaluation_faking_detection \
        --tensor_parallel_size 2 \
        --model_name vllm:Qwen2.5-72B-Instruct \
        --input_file results/${data_name}/${model_name}/mitigation_${system_prompt}.json 

    space_separated_input_file="${space_separated_input_file} results/${data_name}/${model_name}/mitigation_${system_prompt}.json"
done

echo "input file list: ${space_separated_input_file}"

# computing the scores
python -m evaluation.evaluation_faking_computation \
    --input_file_list ${space_separated_input_file} \
    --output_file results/${data_name}/${model_name}/mitigation_ef_evaluation.json
