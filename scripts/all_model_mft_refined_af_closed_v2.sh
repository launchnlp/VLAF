model_list=(
    # openai:gpt-5.4
    bedrock:us.anthropic.claude-sonnet-4-6
)
tensor_parallel_size=1
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
        
        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong_v2 \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --condition \
            --output_file results/MFT_refined_50/${value}/${model_name}_v2_True.json \
            --enable_thinking

        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong_v2 \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --output_file results/MFT_refined_50/${value}/${model_name}_v2_False.json \
            --enable_thinking

    done

done