import os
import torch
import psutil
import numpy as np
from tqdm import tqdm
from typing import List
from transformers import AutoModelForCausalLM, AutoTokenizer

def vanilla_hidden_state_extraction(
    llm: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: List[str],
    batch_size: int = 8,
    append_last_only: bool = False,
    logging: bool = False
) -> List[List[torch.Tensor]]:
    '''
        Extracting hidden states from the model for each layer and token position
        output[sample_index][layer_index] = tensor of shape (seq_len, hidden_size)
    '''

    # storing all the hidden states
    all_hidden_states = [[] for _ in range(len(texts))]
    hidden_layers_list = range(0, llm.config.num_hidden_layers + 1)

    # process for monitoring CPU memory
    proc = psutil.Process(os.getpid())

    # processing in batches
    with torch.no_grad():
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]

            # tokenize all the inputs with left padding
            encoded_inputs = tokenizer(
                batch_texts, 
                padding=True,
                truncation=True,
                return_tensors='pt',
                padding_side='left'
            )
            encoded_inputs = encoded_inputs.to(llm.device)
            
            if logging:
                # GPU memory (GB)
                curr_gpu = torch.cuda.memory_allocated(llm.device) / 1024**3
                peak_gpu = torch.cuda.max_memory_allocated(llm.device) / 1024**3

                # CPU memory (GB) – resident set size
                cpu_rss = proc.memory_info().rss / 1024**3

                print(
                    f"batch {i // batch_size}: "
                    f"CPU={cpu_rss:.2f} GB, GPU curr={curr_gpu:.2f} GB, peak={peak_gpu:.2f} GB"
                )

            # getting the model outputs with hidden states
            out = llm(**encoded_inputs, output_hidden_states=True)
            attention_mask = encoded_inputs['attention_mask']

            # iterating through each hidden layer and each sample
            for layer_idx in hidden_layers_list:
                for sample_idx in range(len(batch_texts)):
                    seq_len = attention_mask[sample_idx].sum().item()
                    if not append_last_only:
                        hidden_state = out.hidden_states[layer_idx][sample_idx, -seq_len:, :].cpu()
                    else:
                        hidden_state = out.hidden_states[layer_idx][sample_idx, -1, :].unsqueeze(0).cpu()
                    all_hidden_states[i + sample_idx].append(hidden_state)

    return all_hidden_states


def layer_last_token_extractor(
    all_hidden_states: List[List[torch.Tensor]],
    layer_idx: int
) -> np.ndarray:
    '''
        Extracting the last-token hidden states from a specific layer and presenting them as a numpy array with dimensions (num_samples, hidden_size)
    '''

    # for collecting the states
    layer_last_token_states = []

    # iterating through all samples
    for sample_all_hidden_states in all_hidden_states:
        
        # getting the last token hidden state for the specified layer
        layer_hidden_state = sample_all_hidden_states[layer_idx][-1, :]
        layer_last_token_states.append(layer_hidden_state.numpy())

    return np.vstack(layer_last_token_states)