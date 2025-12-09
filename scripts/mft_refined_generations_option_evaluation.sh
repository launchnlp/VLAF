model_list=(
    openai:gpt-5
    vllm:olmo2-13b-instruct
    vllm:Qwen2.5-14B-Instruct
    vllm:Qwen3-14B
)
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT evaluation using GPT-5 for extracting the options for model: ${model_name} on value: ${value}"

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_generations/${value}/${model_name}.json

        python -m evaluation.pipeline_evaluation \
            --system_prompt option_selection \
            --input_file results/MFT_refined_generations/${value}/${model_name}_reverse.json
    done
done