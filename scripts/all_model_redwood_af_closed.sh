reasoning_model_list=(
    # openai:gpt-5.4
    # bedrock:us.anthropic.claude-sonnet-4-6
    # bedrock:us.anthropic.claude-3-7-sonnet-20250219-v1:0
)
non_reasoning_model_list=(
    # bedrock:openai.gpt-oss-120b-1:0
    # bedrock:openai.gpt-oss-20b-1:0
    openai:gpt-4o
)

tensor_parallel_size=1

# run MFT inference for all the models
for model_name in "${reasoning_model_list[@]}"; do

    # iterating through all the datasets
    echo "Running MFT inference for model: ${model_name}"
    
    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --condition \
        --output_file results/redwood_helpful/${model_name}_True.json \
        --enable_thinking

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --output_file results/redwood_helpful/${model_name}_False.json \
        --enable_thinking


    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --condition \
        --output_file results/redwood_animal_welfare/${model_name}_True.json \
        --enable_thinking

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --output_file results/redwood_animal_welfare/${model_name}_False.json \
        --enable_thinking

done


# running inference for non-reasoning models
for model_name in "${non_reasoning_model_list[@]}"; do

    # iterating through all the datasets
    echo "Running MFT inference for model: ${model_name}"
    
    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --condition \
        --output_file results/redwood_helpful/${model_name}_True.json

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt helpful_only_noexamples \
        --data redwood_helpful \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --output_file results/redwood_helpful/${model_name}_False.json


    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --condition \
        --output_file results/redwood_animal_welfare/${model_name}_True.json

    python -m inference.pipeline_inference \
        --model_name ${model_name} \
        --tensor_parallel_size ${tensor_parallel_size} \
        --system_prompt animal_welfare_noexamples \
        --data redwood_animal_welfare \
        --temperature 0.9 \
        --batch_size 1 \
        --n 5 \
        --output_file results/redwood_animal_welfare/${model_name}_False.json
done