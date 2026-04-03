values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

approaches=(
    consistency
)

# run MFT reasoning analysis on adapter outputs (free tier only)
# layer indices: olmo2-7b-instruct=15, olmo2-13b-instruct=20, Qwen3-8B=15

for value in "${values[@]}"; do
    for approach in "${approaches[@]}"; do

        # ── olmo2-7b-instruct (layer 15) ─────────────────────────────────────
        echo "Running reasoning analysis for olmo2-7b-instruct, value: ${value}, approach: ${approach}_15"
        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_adapter/${value}/vllm:olmo2-7b-instruct/training_output/olmo2-7b-instruct/activation_mapper/${approach}_15_True.json

        # ── olmo2-13b-instruct (layer 20) ────────────────────────────────────
        echo "Running reasoning analysis for olmo2-13b-instruct, value: ${value}, approach: ${approach}_20"
        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_adapter/${value}/vllm:olmo2-13b-instruct/training_output/olmo2-13b-instruct/activation_mapper/${approach}_20_True.json

        # ── Qwen3-8B (layer 15) ──────────────────────────────────────────────
        echo "Running reasoning analysis for Qwen3-8B, value: ${value}, approach: ${approach}_15"
        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_adapter/${value}/vllm:Qwen3-8B/training_output/Qwen3-8B/activation_mapper/${approach}_15_True.json

    done
done
