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
values=(
    # authority
    care
    fairness
    # loyalty
    # sanctity
)

# run MFT inference for all the adapters
for adapter_name in "${adapter_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running Option based evaluations for adapter: ${adapter_name} on value: ${value} for model: ${model_name}"

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_True.json

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_False.json

        python -m evaluation.option_based_compliance \
            --oversight_file results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_True.json \
            --non_oversight_file "results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_False.json" \
            --key "option_selection_evaluation" \
            --output_file "results/MFT_refined_50_adapter/${value}/${model_name}/${adapter_name}_option_based_compliance.json"
    done

done