reasoning_model_list=(
    # bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
    # bedrock:us.anthropic.claude-sonnet-4-6
    # openai:gpt-5.4
)
non_reasoning_model_list=(
    # bedrock:openai.gpt-oss-120b-1:0
    # bedrock:openai.gpt-oss-20b-1:0
    openai:gpt-4o
)
tensor_parallel_size=1
values=(
    # authority
    # care
    fairness
    # loyalty
    sanctity
)

# run MFT inference for all the models
for model_name in "${reasoning_model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT inference for model: ${model_name} on value: ${value}"
        
        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --condition \
            --output_file results/MFT_refined_50/${value}/${model_name}_True.json \
            --enable_thinking

        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --output_file results/MFT_refined_50/${value}/${model_name}_False.json \
            --enable_thinking

    done

done

# run MFT inference for all the non-reasoning models
for model_name in "${non_reasoning_model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        echo "Running MFT inference for model: ${model_name} on value: ${value}"
        
        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --condition \
            --output_file results/MFT_refined_50/${value}/${model_name}_True.json

        python -m inference.pipeline_inference \
            --model_name ${model_name} \
            --tensor_parallel_size ${tensor_parallel_size} \
            --system_prompt mft_${value}_strong \
            --data mft_refined_${value}_50 \
            --temperature 0.9 \
            --batch_size 1 \
            --output_file results/MFT_refined_50/${value}/${model_name}_False.json

    done

done