target_layers=(
    15
    15
    20
    20
)
model_list=(
    "olmo2-7b-instruct"
    "olmo2-13b-instruct"
    "Qwen3-8B"
    "Qwen3-14B"
)
target_modules_list=(
    'all'
    'qkv'
)


# iterate over the model and layer (1-1 mapping)
for i in "${!model_list[@]}"; do
    model="${model_list[$i]}"
    layer="${target_layers[$i]}"


    # iterate over target modules
    for target_module in "${target_modules_list[@]}"; do
    
    target_directory="/data/inderjeet/af_mitigation/$model/activation_mapper"
    
        echo "Training activation consistency mapper for layer $layer with target module $target_module and model $model"
        accelerate launch --num_processes=1 \
            -m rep_training.activation_mapper_training \
            --approach consistency \
            --model_name "$model" \
            --output_dir "$target_directory/consistency_${layer}_$target_module" \
            --layers_to_transform "$layer" \
            --target_modules "$target_module" \
            --num_epochs 10 \
            --plot

        echo "Training activation redirection mapper for layer $layer with target module $target_module and model $model"
        accelerate launch --num_processes=1 \
            -m rep_training.activation_mapper_training \
            --approach redirection \
            --model_name "$model" \
            --output_dir "$target_directory/redirection_${layer}_$target_module" \
            --layers_to_transform "$layer" \
            --target_modules "$target_module" \
            --num_epochs 10 \
            --plot

    done
    
done