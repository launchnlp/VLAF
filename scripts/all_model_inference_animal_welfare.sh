# creating shell script variables
system_prompt_list=(
    "animal_welfare"
    "animal_welfare_noexamples"
)
model_list=(
    "Qwen3-8B"
    "Qwen3-14B"
    "Llama-3-8B-Instruct"
)
data_list=(
    redwood_animal_welfare
)
tensor_parallel_size=2

# iterating over the different models
for model_name in "${model_list[@]}"; do
    for system_prompt in "${system_prompt_list[@]}"; do
        for data in "${data_list[@]}"; do
            echo "Running inference for model: ${model_name}, system prompt: ${system_prompt}, data: ${data}"

            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt ${system_prompt} \
                --data ${data} \
                --temperature 0.7 \
                --top_p 0.95 \
                --in_training
            
            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt ${system_prompt} \
                --data ${data} \
                --temperature 0.7 \
                --top_p 0.95
        done
    done
done