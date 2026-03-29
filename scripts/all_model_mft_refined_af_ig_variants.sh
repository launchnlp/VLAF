model_list=(
    vllm:Qwen3-8B
    vllm:Qwen3-14B
    vllm:Qwen3-32B
    # openai:gpt-5
    vllm:Qwen2.5-7B-Instruct
    vllm:Qwen2.5-14B-Instruct
    vllm:Qwen2.5-32B-Instruct
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:olmo2-32b-instruct
    # vllm:olmo2-32b-sft
    # vllm:olmo2-32b-dpo
)
tensor_parallel_size=1
values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# instrumental goal variant suffixes
# is = influence seeking, ra = resource acquisition, sp = self-preservation, strong = value preservation
ig_variants=(
    is_1
    is_2
    is_3
    ra_1
    ra_2
    ra_3
    sp_1
    sp_2
    sp_3
    strong_1
    strong_2
    strong_3
)

# run MFT inference for all models across all values and instrumental goal variants
for model_name in "${model_list[@]}"; do

    for value in "${values[@]}"; do

        for ig_variant in "${ig_variants[@]}"; do
            echo "Running MFT inference for model: ${model_name}, value: ${value}, ig_variant: ${ig_variant}"

            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_${value}_${ig_variant} \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --condition \
                --output_file results/MFT_refined_50/${value}/${model_name}_${ig_variant}_True.json

            python -m inference.pipeline_inference \
                --model_name ${model_name} \
                --tensor_parallel_size ${tensor_parallel_size} \
                --system_prompt mft_${value}_${ig_variant} \
                --data mft_refined_${value}_50 \
                --temperature 0.9 \
                --output_file results/MFT_refined_50/${value}/${model_name}_${ig_variant}_False.json

        done

    done

done
