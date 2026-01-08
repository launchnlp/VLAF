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
    -0.01
    -0.05
    0.05
    0.01
    # -2.0
    # -1.0
    # 1.0
    # 2.0
    # 0.0
)
layers=(
    all
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
                    
                    echo "Running GPT-5 option extraction for model: ${model_name} on value: ${value} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer}"

                    python -m evaluation.pipeline_evaluation \
                        --system_prompt option_selection \
                        --input_file results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_True.json

                    python -m evaluation.pipeline_evaluation \
                        --system_prompt option_selection \
                        --input_file results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_False.json

                    echo "Completed GPT-5 option extraction for model: ${model_name} on value: ${value} with vector file: ${vector_file}, strength: ${strength_value}, layer: ${layer} - Now running option based compliance computation"

                    python -m evaluation.option_based_compliance \
                        --oversight_file results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_True.json \
                        --non_oversight_file "results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_False.json" \
                        --key "option_selection_evaluation" \
                        --output_file "results/MFT_refined_steering/${value}/${model_name}_${vector_file}_${layer}_${strength_value}_option_based_compliance.json"
                    
                done

            done

        done

    done

done