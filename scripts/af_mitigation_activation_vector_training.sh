model_list=(
    # olmo2-7b-instruct
    # olmo2-13b-instruct
    olmo2-32b-instruct
    # Qwen3-8B
    Qwen3-14B
    Qwen3-32B

)
approach_list=(
    diffmean
    pca
)
dataset_list=(
    sorry_bench_sa
    wmdp_sa
)

# iterating over the models, approaches, and datasets
for model_name in "${model_list[@]}"; do
    for approach in "${approach_list[@]}"; do
        for dataset_name in "${dataset_list[@]}"; do
            echo "Processing model: $model_name, approach: $approach, dataset: $dataset_name"
            python -m rep_extraction.rep_extractor_sa_data \
                --model_name "$model_name" \
                --approach "$approach" \
                --data_name "$dataset_name" \
                --output_file "training_output/${model_name}/${approach}_${dataset_name}.gguf"
        done
    done
done