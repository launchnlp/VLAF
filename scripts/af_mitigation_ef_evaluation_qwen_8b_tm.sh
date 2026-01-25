# Script for testing the presence of evaluation faking

model_name="vllm:Qwen3-8B"
adapter_list=(
    training_output/Qwen3-8B/activation_mapper/consistency_15_all
    training_output/Qwen3-8B/activation_mapper/consistency_15_qkv
    training_output/Qwen3-8B/activation_mapper/redirection_15_all
    training_output/Qwen3-8B/activation_mapper/redirection_15_qkv
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
data_name="strong_reject_50"

# running the inference for each adapter
for adapter_name in "${adapter_list[@]}"; do

    # resetting the input file list
    space_separated_input_file=""

    # iterating over all the system prompts
    for system_prompt in "${system_prompt_list[@]}"; do
        echo "Running EF evaluation for model: ${model_name} with adapter: ${adapter_name} and system prompt: ${system_prompt}"
        python -m evaluation.pipeline_evaluation \
            --system_prompt evaluation_faking_detection \
            --tensor_parallel_size 2 \
            --model_name vllm:Qwen2.5-72B-Instruct \
            --input_file results/${data_name}/${model_name}/$(basename ${adapter_name})_${system_prompt}.json

        space_separated_input_file="${space_separated_input_file} results/${data_name}/${model_name}/$(basename ${adapter_name})_${system_prompt}.json"
    done

    echo "input file list: ${space_separated_input_file}"

    # computing the scores
    python -m evaluation.evaluation_faking_computation \
        --input_file_list ${space_separated_input_file} \
        --output_file results/${data_name}/${model_name}/$(basename ${adapter_name})_ef_evaluation.json

done