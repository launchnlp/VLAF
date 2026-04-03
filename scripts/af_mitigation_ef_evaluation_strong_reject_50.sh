model_list=(
    vllm:olmo2-7b-instruct
    vllm:olmo2-13b-instruct
    vllm:Qwen3-8B
)
approaches=(
    diffmean
    pca
)
datasets=(
    sorry_bench
    wmdp
)
hyperparams=(
    # 3.0
    6.0
    # 12.0
)
system_prompt_list=(
    ef_honeypot_s1
    ef_honeypot_s2
    ef_honeypot_s3
    ef_honeypot_s4
    ef_honeypot_s5
    ef_honeypot_s7
)
data_name="strong_reject_50"

for model_name in "${model_list[@]}"; do

    # set layer count based on model
    if [[ "${model_name}" == *"13b"* ]]; then
        n=20
    else
        n=15
    fi

    for approach in "${approaches[@]}"; do
        for dataset in "${datasets[@]}"; do
            for hp in "${hyperparams[@]}"; do

                prefix="${approach}_${dataset}_sa.gguf_${n}_${hp}"
                space_separated_input_file=""

                for system_prompt in "${system_prompt_list[@]}"; do
                    input_file="results/${data_name}/${model_name}/${prefix}_${system_prompt}.json"
                    echo "Running EF evaluation for model: ${model_name}, approach: ${prefix}, system prompt: ${system_prompt}"

                    python -m evaluation.pipeline_evaluation \
                        --system_prompt evaluation_faking_detection \
                        --tensor_parallel_size 1 \
                        --model_name vllm:Qwen2.5-32B-Instruct \
                        --input_file ${input_file}

                    space_separated_input_file="${space_separated_input_file} ${input_file}"
                done

                echo "Computing EF scores for: ${model_name} / ${prefix}"
                python -m evaluation.evaluation_faking_computation \
                    --input_file_list ${space_separated_input_file} \
                    --output_file results/${data_name}/${model_name}/${prefix}_ef_evaluation.json

            done
        done
    done

done
