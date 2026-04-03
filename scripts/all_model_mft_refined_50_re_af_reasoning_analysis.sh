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
    pca_sorry_bench_sa.gguf
    pca_wmdp_sa.gguf
)

strength=6.0

# run MFT reasoning analysis on steer vector outputs (free tier only, strength=6.0)
# layer indices: olmo2-7b-instruct=15, olmo2-13b-instruct=20, Qwen3-8B=15

for value in "${values[@]}"; do
    for vector_file in "${vector_files[@]}"; do

        # ── olmo2-7b-instruct (layer 15) ─────────────────────────────────────
        echo "Running reasoning analysis for olmo2-7b-instruct, value: ${value}, vector: ${vector_file}"

        # report the progress
        echo "Processing value: ${value}, vector: ${vector_file}, strength: ${strength}, model: olmo2-7b-instruct, layer: 15" >> results/MFT_refined_50_re/reasoning_analysis_progress_1.log

        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_re/${value}/vllm:olmo2-7b-instruct_${vector_file}_15_${strength}_True.json

        # ── olmo2-13b-instruct (layer 20) ────────────────────────────────────
        echo "Running reasoning analysis for olmo2-13b-instruct, value: ${value}, vector: ${vector_file}"

        # report the progress
        echo "Processing value: ${value}, vector: ${vector_file}, strength: ${strength}, model: olmo2-13b-instruct, layer: 20" >> results/MFT_refined_50_re/reasoning_analysis_progress_1.log

        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_re/${value}/vllm:olmo2-13b-instruct_${vector_file}_20_${strength}_True.json

        # ── Qwen3-8B (layer 15) ──────────────────────────────────────────────
        echo "Running reasoning analysis for Qwen3-8B, value: ${value}, vector: ${vector_file}"

        # report the progress
        echo "Processing value: ${value}, vector: ${vector_file}, strength: ${strength}, model: Qwen3-8B, layer: 15" >> results/MFT_refined_50_re/reasoning_analysis_progress_1.log

        python -m evaluation.pipeline_evaluation \
            --system_prompt mft_reasoning_analysis \
            --tensor_parallel_size 1 \
            --model_name vllm:Qwen2.5-32B-Instruct \
            --input_file results/MFT_refined_50_re/${value}/vllm:Qwen3-8B_${vector_file}_15_${strength}_True.json

    done
done
