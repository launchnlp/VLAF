model_name="vllm:olmo2-7b-instruct-pt"
adapter_list=(
    # training_output/olmo2-7b-instruct/activation_mapper/test
    # training_output/olmo2-7b-instruct/activation_mapper/consistency_5
    # training_output/olmo2-7b-instruct/activation_mapper/redirection_5
    # training_output/olmo2-7b-instruct/activation_mapper/consistency_10
    # training_output/olmo2-7b-instruct/activation_mapper/redirection_10
    training_output/olmo2-7b-instruct/activation_mapper/consistency_15
    training_output/olmo2-7b-instruct/activation_mapper/redirection_15
    # training_output/olmo2-7b-instruct/activation_mapper/consistency_20
    # training_output/olmo2-7b-instruct/activation_mapper/redirection_20
    # training_output/olmo2-7b-instruct/activation_mapper/consistency_25
    # training_output/olmo2-7b-instruct/activation_mapper/redirection_25
)
tensor_parallel_size=1
values=(
    care
    # fairness
    # authority
    # loyalty
    # sanctity
)

# run MFT inference for all the models
for adapter_name in "${adapter_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT inference for adapter: ${adapter_name}, model: ${model_name} on value: ${value}"
        
        python -m rep_training.inference \
            --model_name ${model_name} \
            --adapter_path ${adapter_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_minimal \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --condition \
            --output_file results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_True.json

        python -m rep_training.inference \
            --model_name ${model_name} \
            --adapter_path ${adapter_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_minimal \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --output_file results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_False.json

    done

done