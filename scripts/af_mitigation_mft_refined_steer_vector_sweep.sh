model_list=(
    olmo2-7b-instruct
    olmo2-13b-instruct
    olmo2-32b-instruct
    Qwen3-8B
    Qwen3-14B
    Qwen3-32B
)

# Max layer (num_hidden_layers) per model, used to bound multiples-of-5 iteration
# olmo2-7b: 32, olmo2-13b: 40, olmo2-32b: 64
# Qwen3-8B: 36, Qwen3-14B: 40, Qwen3-32B: 64
declare -A model_max_layer=(
    [olmo2-7b-instruct]=30
    [olmo2-13b-instruct]=40
    [olmo2-32b-instruct]=60
    [Qwen3-8B]=35
    [Qwen3-14B]=40
    [Qwen3-32B]=60
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
    diffmean_sorry_bench_sa.gguf
    diffmean_wmdp_sa.gguf
    pca_sorry_bench_sa.gguf
    pca_wmdp_sa.gguf
)

# intervention strengths: 3, 6, 12, 24
strength_list=(3 6 12 24)

for model_name in "${model_list[@]}"; do

    max_layer="${model_max_layer[$model_name]}"

    # iterate over layers in multiples of 5
    for layer in $(seq 5 5 "$max_layer"); do

        # iterate over strength values
        for strength_value in "${strength_list[@]}"; do

            # iterate over vector files
            for vector_file in "${vector_files[@]}"; do

                # iterate over MFT values
                for value in "${values[@]}"; do

                    echo "Running MFT inference for model: ${model_name} | layer: ${layer} | strength: ${strength_value} | vector: ${vector_file} | value: ${value}"

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

done
