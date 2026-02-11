import torch
import argparse
from safetensors.torch import load_file
from rep_extraction.grassman_analyzer import lora_subspace_similarity

def pipeline_lora_grassman_analyzer(
    train_location_1: str,
    train_location_2: str,
    save_location: str
) -> None:
    '''
        Pipeline for analyzing the trained lora parameters for the sorry bench and wmdp datasets using grassmannian analysis
    '''

    # loading the trained lora parameters using safetensors
    lora_params_1 = load_file(train_location_1)
    lora_params_2 = load_file(train_location_2)

    # testing the similarities for a key: base_model.model.model.layers.15.mlp.down_proj.lora_A.weight
    weight_1 = lora_params_1['base_model.model.model.layers.15.mlp.up_proj.lora_B.weight']
    weight_2 = lora_params_2['base_model.model.model.layers.15.mlp.up_proj.lora_B.weight']
    similarity = lora_subspace_similarity(
        rep_1=weight_1.float(),
        rep_2=weight_2.float()
    )

    

    import pdb; pdb.set_trace()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyzing the trained lora parameters for the sorry bench and wmdp datasets using grassmannian analysis')
    parser.add_argument('--train_location_1', type=str, default='training_output/olmo2-7b-instruct/activation_mapper/consistency_15/adapter_model.safetensors')
    parser.add_argument('--train_location_2', type=str, default='training_output/olmo2-7b-instruct/activation_mapper_wmdp/consistency_15/adapter_model.safetensors')
    parser.add_argument('--save_location', type=str, default='plotting/trained_output_grassmanian_results_olmo2-7b-instruct.pt')
    args = parser.parse_args()

    pipeline_lora_grassman_analyzer(
        train_location_1=args.train_location_1,
        train_location_2=args.train_location_2,
        save_location=args.save_location
    )