model_list=(
    olmo2-7b-instruct
    Qwen3-8B
    # olmo2-13b-instruct
    # olmo2-32b-instruct
    # Qwen3-14B
    # Qwen3-32B
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

    for layer in $(seq 5 5 "$max_layer"); do

        for strength_value in "${strength_list[@]}"; do

            for vector_file in "${vector_files[@]}"; do

                for value in "${values[@]}"; do

                    oversight_file="results/MFT_refined_50_re/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_True.json"
                    non_oversight_file="results/MFT_refined_50_re/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_False.json"
                    compliance_file="results/MFT_refined_50_re/${value}/vllm:${model_name}_${vector_file}_${layer}_${strength_value}_option_based_compliance.json"

                    echo "Evaluating: model=${model_name} | layer=${layer} | strength=${strength_value} | vector=${vector_file} | value=${value}"

                    python -m evaluation.pipeline_evaluation \
                        --system_prompt option_selection \
                        --tensor_parallel_size 1 \
                        --model_name vllm:Qwen2.5-32B-Instruct \
                        --input_file "${oversight_file}" \
                        --no_thinking

                    python -m evaluation.pipeline_evaluation \
                        --system_prompt option_selection \
                        --tensor_parallel_size 1 \
                        --model_name vllm:Qwen2.5-32B-Instruct \
                        --input_file "${non_oversight_file}" \
                        --no_thinking

                    python -m evaluation.option_based_compliance \
                        --oversight_file "${oversight_file}" \
                        --non_oversight_file "${non_oversight_file}" \
                        --key "option_selection_evaluation" \
                        --output_file "${compliance_file}"

                done

            done

        done

    done

done
