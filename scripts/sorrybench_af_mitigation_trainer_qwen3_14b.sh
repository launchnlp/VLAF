target_layers=(
    5
    10
    15
    20
    25
    30
    35
)
model="Qwen3-14B"
target_directory="/data/inderjeet/af_mitigation/$model/activation_mapper"

for layer in "${target_layers[@]}"; do
    
    echo "Training activation consistency mapper for layer $layer"
    accelerate launch --num_processes=2 \
        -m rep_training.activation_mapper_training \
        --approach consistency \
        --model_name "$model" \
        --output_dir "$target_directory/consistency_$layer" \
        --layers_to_transform "$layer" \
        --num_epochs 20 \
        --plot

    echo "Training activation redirection mapper for layer $layer"
    accelerate launch --num_processes=2 \
        -m rep_training.activation_mapper_training \
        --approach redirection \
        --model_name "$model" \
        --output_dir "$target_directory/redirection_$layer" \
        --layers_to_transform "$layer" \
        --num_epochs 20 \
        --plot
    
done