model_list=(
    vllm:olmo2-7b-instruct
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
    results/af_representation/olmo2-7b-instruct_mft_refined_care_diffmean.gguf
    results/af_representation/olmo2-7b-instruct_mft_refined_care_pca.gguf
)
strength=(
    -2.0
    -1.0
    1.0
    2.0
)
layers=(
    5
    10
    15
    20
)

# run MFT inference for all the models
for model_name in "${model_list[@]}"; do

    # iterating through all the datasets
    for value in "${values[@]}"; do
        
        # iterating through all the vector files
        for vector_file in "${vector_files[@]}"; do
            
            # iterating through all the strengths
            for strength_value in "${strength[@]}"; do
                
                # iterating through all the layers
                for layer in "${layers[@]}"; do
                    
                    echo "Running MFT inference for model: ${model_name} on value: ${value} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer}"
                    
                    python -m rep_extraction.pipeline_inference \
                        --model_name ${model_name} \
                        --tensor_parallel_size ${tensor_parallel_size} \
                        --system_prompt mft_${value}_strong \
                        --data mft_refined_${value}_final \
                        --sample_size 30 \
                        --temperature 0.9 \
                        --steering_vector_path ${vector_file} \
                        --steering_strength ${strength_value} \
                        --application_layer ${layer} \
                        --condition \
                        --output_file results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_True.json

                    python -m rep_extraction.pipeline_inference \
                        --model_name ${model_name} \
                        --tensor_parallel_size ${tensor_parallel_size} \
                        --system_prompt mft_${value}_strong \
                        --data mft_refined_${value}_final \
                        --sample_size 30 \
                        --temperature 0.9 \
                        --steering_vector_path ${vector_file} \
                        --steering_strength ${strength_value} \
                        --application_layer ${layer} \
                        --output_file results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_False.json
                done

            done

        done

    done

done