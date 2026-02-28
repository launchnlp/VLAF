model_list=(
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:Qwen3-8B
)
layer_list=(
    15
    20
    15
)
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

# run MFT inference for all the models
for i in "${!model_list[@]}"; do

    # get the corresponding layer number for the model
    layer_num=${layer_list[$i]}
    model_name=${model_list[$i]}

    # iterating over all the vector files
    for vector_file in "${vector_files[@]}"; do

        # iterating over all the strength values
        for strength in "${strength_list[@]}"; do

            # iterating through all the datasets
            for value in "${values[@]}"; do
                echo "Running Option based evaluations for model: ${model_name} on value: ${value} with vector file: ${vector_file} and strength: ${strength} with layer num: ${layer_num}"

                python -m evaluation.pipeline_evaluation \
                    --system_prompt option_selection \
                    --tensor_parallel_size 2 \
                    --model_name vllm:Qwen2.5-72B-Instruct \
                    --input_file results/MFT_refined_50_re/${value}/${model_name}_${vector_file}_${layer_num}_${strength}_True.json

                python -m evaluation.pipeline_evaluation \
                    --system_prompt option_selection \
                    --tensor_parallel_size 2 \
                    --model_name vllm:Qwen2.5-72B-Instruct \
                    --input_file results/MFT_refined_50_re/${value}/${model_name}_${vector_file}_${layer_num}_${strength}_False.json

                python -m evaluation.option_based_compliance \
                    --oversight_file results/MFT_refined_50_re/${value}/${model_name}_${vector_file}_${layer_num}_${strength}_True.json \
                    --non_oversight_file "results/MFT_refined_50_re/${value}/${model_name}_${vector_file}_${layer_num}_${strength}_False.json" \
                    --key "option_selection_evaluation" \
                    --output_file "results/MFT_refined_50_re/${value}/${model_name}_${vector_file}_${layer_num}_${strength}_option_based_compliance.json"
            done

        done

    done

done