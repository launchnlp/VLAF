model_list=(
    vllm:Qwen3-14B
    openai:gpt-5
    vllm:Qwen2.5-14B-Instruct
    vllm:olmo2-13b-instruct
)
tensor_parallel_size=4
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
        echo "Running MFT inference for model: ${model_name} on value: ${value}"
        
        python -m data_generation.scenario_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_no_faking \
            --data mft_refined_${value} mft_refined_${value}_generations \
            --temperature 0.9 \
            --reverse \
            --output_file results/MFT_refined_generations/${value}/${model_name}_reverse.json

        python -m data_generation.scenario_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_no_faking \
            --data mft_refined_${value} mft_refined_${value}_generations \
            --temperature 0.9 \
            --output_file results/MFT_refined_generations/${value}/${model_name}.json

    done

done