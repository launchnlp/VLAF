target_layers=(
    15
    20
    15
)
model_list=(
    "olmo2-7b-instruct"
    "olmo2-13b-instruct"
    "Qwen3-8B"
)
system_prompt_list=(
    ef_honeypot_s1
    ef_honeypot_s2
    ef_honeypot_s3
    ef_honeypot_s4
    ef_honeypot_s5
    ef_honeypot_s7
)
vector_files=(
    # diffmean_sorry_bench_sa.gguf
    # diffmean_wmdp_sa.gguf
    # pca_sorry_bench_sa.gguf
    pca_wmdp_sa.gguf
)
strength_list=(
    3.0
    6.0
    12.0
)
layer_list=(
    15
    20
    15
)
data_name="strong_reject_50"
tensor_parallel_size=1

# iterate over the model and layer 
for i in "${!model_list[@]}"; do

    # get the model name and layer from the respective lists
    model_name="${model_list[$i]}"
    layer="${layer_list[$i]}"

    # iterate over the strength values
    for strength_value in "${strength_list[@]}"; do
        
        # iterate over the vector files
        for vector_file in "${vector_files[@]}"; do

            # iterate over the system prompts
            for system_prompt in "${system_prompt_list[@]}"; do
                
                echo "Running Ef inference: ${model_name} with system prompt: ${system_prompt} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer}"
                
                python -m rep_extraction.pipeline_inference \
                    --model_name vllm:${model_name} \
                    --tensor_parallel_size ${tensor_parallel_size} \
                    --system_prompt ${system_prompt} \
                    --data ${data_name} \
                    --temperature 0.9 \
                    --steering_vector_path training_output/${model_name}/${vector_file} \
                    --steering_strength ${strength_value} \
                    --application_layer ${layer} \
                    --output_file results/${data_name}/vllm:${model_name}/${vector_file}_${layer}_${strength_value}_${system_prompt}.json

            done

        done

    done

done