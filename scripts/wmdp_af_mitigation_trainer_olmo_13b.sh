target_layers=(
    20
    5
    10
    15
    25
    30
    35
)
model="olmo2-13b-instruct"
target_directory="training_output/$model/activation_mapper_wmdp"
data_name="wmdp_sa"

for layer in "${target_layers[@]}"; do
    
    echo "Training activation consistency mapper for layer $layer"
    python \
        -m rep_training.activation_mapper_training \
        --approach consistency \
        --model_name "$model" \
        --output_dir "$target_directory/consistency_$layer" \
        --data_name "$data_name" \
        --layers_to_transform "$layer" \
        --num_epochs 10 \
        --plot

    echo "Training activation redirection mapper for layer $layer"
    python \
        -m rep_training.activation_mapper_training \
        --approach redirection \
        --model_name "$model" \
        --output_dir "$target_directory/redirection_$layer" \
        --data_name "$data_name" \
        --layers_to_transform "$layer" \
        --num_epochs 10 \
        --plot
    
done