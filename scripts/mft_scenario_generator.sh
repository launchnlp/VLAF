values=(
    authority
    care
    fairness
    loyalty
    sanctity
)

# run scenario_generator.py for all the values
for value in "${values[@]}"; do
    echo "Generating MFT scenarios for value: ${value}"

    python -m data_generation.scenario_generator \
        --system_prompt mft_scenario_generation_${value} \
        --data mft_refined_${value} \
        --output_file data/mft_refined/${value}_generations.json
done
