'''
    Here, we will be training two complementary activation mapping approaches:
    1. Activation consistency: For each example, the last token representations corresponding to the different oversight conditions must have minimal distance.
    2. Activation redirection: For each example, the last token representations would be redirected with an appropriate redirection vector to minimize the distance between the different oversight conditions.
'''
import os
import torch
from tqdm import tqdm
from torch.optim import AdamW
from accelerate import Accelerator
from utils.vllm_utils import model_dictionary
from peft import LoraConfig, TaskType, get_peft_model
from transformers import HfArgumentParser, AutoModelForCausalLM, AutoTokenizer, get_scheduler
from rep_training.argument import ModelArguments, LoraArguments, TrainingArguments, ActivationMapperArguments
from rep_training.data_processor import process_sorry_bench_sa_data, tokenize_dataset, dataset_to_dataloader, associate_representations
from rep_training.train_utils import last_token_rep_extractor, activation_consistency_loss, activation_redirection_loss, compute_original_representations, plot_umap_2d, numpify_representations

def pipeline_activation_mapper_trainer(
    model_args: ModelArguments,
    lora_args: LoraArguments,
    training_args: TrainingArguments,
    activation_mapper_args: ActivationMapperArguments
) -> None:
    '''
        Pipeline for training the activation mapper
    '''

    # initializing accelerator for distributed training
    accelerator = Accelerator()
    
    # loading the model
    if model_args.model_name not in model_dictionary:
        raise ValueError(f"Model '{model_args.model_name}' not found in model dictionary.")
    model_info = model_dictionary[model_args.model_name]
    model = AutoModelForCausalLM.from_pretrained(
        model_info['model'],
        device_map='auto'
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_info['model']
    )

    # loading the dataset
    dataset_dict = process_sorry_bench_sa_data(
        tokenizer,
        random_seed=42,
        evaluation_split=0.2
    )
    tokenized_dataset_dict = tokenize_dataset(
        dataset_dict, tokenizer=tokenizer
    )

    # applying representations if approach is redirection
    if activation_mapper_args.approach == 'redirection':
        tokenized_dataset_dict = associate_representations(
            tokenized_dataset_dict,
            model,
            tokenizer,
            lora_args.layers_to_transform + 1
        )
    
    # creating the dataloader - representations created in the previous step would be appropriately collated
    train_dataloader, eval_dataloader = dataset_to_dataloader(
        tokenized_dataset_dict,
        tokenizer=tokenizer,
        batch_size=training_args.batch_size
    )

    # apply LoRA to the model
    if lora_args.target_modules == 'all':
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
    elif lora_args.target_modules == 'mlp':
        target_modules = ['up_proj', 'down_proj', 'gate_proj']
    elif lora_args.target_modules == 'qkv':
        target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj']
    lora_config = LoraConfig(
        r=lora_args.lora_r,
        lora_alpha=lora_args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=lora_args.lora_dropout,
        layers_to_transform=[lora_args.layers_to_transform],
        bias='none',
        task_type=TaskType.CAUSAL_LM
    )
    lora_model = get_peft_model(model, lora_config)
    
    # preparing optimizer and scheduler and accelerator
    num_update_steps_per_epoch = int(
        len(train_dataloader) / training_args.gradient_accumulation_steps
    )
    max_train_steps = training_args.num_epochs * num_update_steps_per_epoch
    num_warmup_steps = int(training_args.warmup_ratio * max_train_steps)
    optimizer = AdamW(
        lora_model.parameters(),
        lr=training_args.learning_rate,
        weight_decay=training_args.weight_decay
    )
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=max_train_steps
    )
    lora_model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        lora_model,
        optimizer,
        lr_scheduler,
        train_dataloader,
        eval_dataloader
    )
    
    # setting up the progress bar
    progress_bar = tqdm(range(max_train_steps), disable=not accelerator.is_local_main_process)
    
    # track best model across epochs
    best_eval_loss = float('inf')

    # training loop
    for epoch in range(training_args.num_epochs):

        # if training_args.plot is True, plot the umap of representations
        if training_args.plot: 

            # representations from the train dataset
            original_oversight_representations, original_non_oversight_representations = compute_original_representations(
                lora_model,
                train_dataloader,
                lora_args.layers_to_transform + 1
            )

            # representations from the eval dataset
            eval_original_oversight_representations, eval_original_non_oversight_representations = compute_original_representations(
                lora_model,
                eval_dataloader,
                lora_args.layers_to_transform + 1
            )

            # saving the umap plots for the representations
            if accelerator.is_main_process:

                # creating the paths
                train_save_path = os.path.join(model_args.output_dir, 'train_plots', f'epoch_{epoch+1}_umap.pdf')
                eval_save_path = os.path.join(model_args.output_dir, 'eval_plots', f'epoch_{epoch+1}_umap.pdf')

                # numpifying the representations
                train_reps_np, train_labels = numpify_representations(
                    original_oversight_representations,
                    original_non_oversight_representations
                )
                eval_reps_np, eval_labels = numpify_representations(
                    eval_original_oversight_representations,
                    eval_original_non_oversight_representations
                )

                # plotting the umap plots
                plot_umap_2d(
                    train_reps_np,
                    train_labels,
                    title=f'Train UMAP Epoch {epoch+1}',
                    save_path=train_save_path
                )
                plot_umap_2d(
                    eval_reps_np,
                    eval_labels,
                    title=f'Eval UMAP Epoch {epoch+1}',
                    save_path=eval_save_path
                )

        # training
        lora_model.train()
        for index, batch in enumerate(train_dataloader):

            # getting the model outputs for non_oversight and oversight conditions
            oversight_representations = last_token_rep_extractor(
                lora_model,
                batch['oversight_input_ids'],
                batch['oversight_attention_mask'],
                lora_args.layers_to_transform + 1
            )
            non_oversight_representations = last_token_rep_extractor(
                lora_model,
                batch['non_oversight_input_ids'],
                batch['non_oversight_attention_mask'],
                lora_args.layers_to_transform + 1
            )

            # computing the loss
            if activation_mapper_args.approach == 'consistency':
                loss = activation_consistency_loss(oversight_representations, non_oversight_representations)
                loss /= training_args.gradient_accumulation_steps
            
            # redirection loss involves anchor representations
            elif activation_mapper_args.approach == 'redirection':

                # getting the anchor representations
                anchor_representations = batch['representations_{}'.format(lora_args.layers_to_transform + 1)]

                loss = activation_redirection_loss(
                    oversight_representations,
                    non_oversight_representations,
                    anchor_representations
                )
                loss /= training_args.gradient_accumulation_steps

            # accelerator steps
            accelerator.backward(loss)

            # gradient accumulation
            if (index + 1) % training_args.gradient_accumulation_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)


        # evaluation loop
        lora_model.eval()
        total_eval_loss = 0.0
        num_eval_batches = 0
        
        with torch.no_grad():
            for batch in eval_dataloader:
                
                # getting the model outputs for non_oversight and oversight conditions
                oversight_representations = last_token_rep_extractor(
                    lora_model,
                    batch['oversight_input_ids'],
                    batch['oversight_attention_mask'],
                    lora_args.layers_to_transform + 1
                )
                non_oversight_representations = last_token_rep_extractor(
                    lora_model,
                    batch['non_oversight_input_ids'],
                    batch['non_oversight_attention_mask'],
                    lora_args.layers_to_transform + 1
                )

                # computing the loss
                if activation_mapper_args.approach == 'consistency':
                    loss = activation_consistency_loss(oversight_representations, non_oversight_representations)
                
                # redirection loss involves anchor representations
                elif activation_mapper_args.approach == 'redirection':
                    anchor_representations = batch['representations_{}'.format(lora_args.layers_to_transform + 1)]
                    loss = activation_redirection_loss(
                        oversight_representations,
                        non_oversight_representations,
                        anchor_representations
                    )

                # accumulate loss across batches
                total_eval_loss += accelerator.gather(loss).mean().item()
                num_eval_batches += 1
        
        # compute average evaluation loss for this epoch
        avg_eval_loss = total_eval_loss / num_eval_batches
        
        # log evaluation results
        if accelerator.is_main_process:
            print(f"Epoch {epoch+1}: Avg Eval Loss = {avg_eval_loss:.4f}")
        
        # save model if this is the best epoch so far
        if avg_eval_loss < best_eval_loss:
            best_eval_loss = avg_eval_loss
            
            # waiting for all processes to sync
            accelerator.wait_for_everyone()
            
            # save the best model (only on main process)
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(lora_model)
                unwrapped_model.save_pretrained(model_args.output_dir)
                tokenizer.save_pretrained(model_args.output_dir)
                print(f"✓ Saved new best model with loss {best_eval_loss:.4f}")
    

if __name__ == '__main__':
    parser = HfArgumentParser((
        ModelArguments,
        LoraArguments,
        TrainingArguments,
        ActivationMapperArguments
    ))
    model_args, lora_args, training_args, activation_mapper_args = parser.parse_args_into_dataclasses()
    pipeline_activation_mapper_trainer(
        model_args,
        lora_args,
        training_args,
        activation_mapper_args
    )