model_list=(
    olmo2-7b-instruct
    olmo2-13b-instruct
    Qwen3-8B
)
tensor_parallel_size=1
values=(
    authority
    care
    fairness
    loyalty
    sanctity
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

# iterate over the model and layer 
for i in "${!model_list[@]}"; do

    # get the model name and layer from the respective lists
    model_name="${model_list[$i]}"
    layer="${layer_list[$i]}"

    # iterate over the strength values
    for strength_value in "${strength_list[@]}"; do
        
        # iterate over the vector files
        for vector_file in "${vector_files[@]}"; do
            
            # iterate over the values
            for value in "${values[@]}"; do
                
                echo "Running MFT inference for model: ${model_name} on value: ${value} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer}"
                
                python -m rep_extraction.pipeline_inference \
                    --model_name vllm:${model_name} \
                    --tensor_parallel_size ${tensor_parallel_size} \
                    --system_prompt mft_${value}_strong \
                    --data mft_refined_${value}_50 \
                    --temperature 0.9 \
                    --steering_vector_path training_output/${model_name}/${vector_file} \
                    --steering_strength ${strength_value} \
                    --application_layer ${layer} \
                    --condition \
                    --output_file results/MFT_refined_50_re/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_True.json

                python -m rep_extraction.pipeline_inference \
                    --model_name vllm:${model_name} \
                    --tensor_parallel_size ${tensor_parallel_size} \
                    --system_prompt mft_${value}_strong \
                    --data mft_refined_${value}_50 \
                    --temperature 0.9 \
                    --steering_vector_path training_output/${model_name}/${vector_file} \
                    --steering_strength ${strength_value} \
                    --application_layer ${layer} \
                    --output_file results/MFT_refined_50_re/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_False.json
            done

        done

    done

done