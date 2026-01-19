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
    training_output/olmo2-7b-instruct/activation_mapper/test
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


# running the inference first for the base model without any adapter
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

# run MFT inference for all the models
for adapter_name in "${adapter_list[@]}"; do

    # iterating over all the system prompts
    for system_prompt in "${system_prompt_list[@]}"; do
        echo "Running EF tester for model: ${model_name} with adapter: ${adapter_name} and system prompt: ${system_prompt}"
        python -m rep_training.inference \
            --model_name ${model_name} \
            --adapter_path ${adapter_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt ${system_prompt} \
            --data ${data_name} \
            --temperature 0.9 \
            --output_file results/${data_name}/${model_name}/$(basename ${adapter_name})_${system_prompt}.json
    done

done