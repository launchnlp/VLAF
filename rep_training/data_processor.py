import random
from typing import Any, Union, Tuple
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from datasets import Dataset, DatasetDict
from data.dataloader import load_sorry_bench_sa_data
from custom_typing.scenario import SorryBenchSAScenario

def convert_to_model_input(
    system_prompt: str,
    user_prompt: str,
    response: Union[str, None],
    tokenizer: AutoTokenizer,
    add_generation_prompt: bool = False
) -> str:
    '''
        Converting the data into model input format
    '''
    
    # creating the message object
    if response is not None:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt},
            {'role': 'assistant', 'content': response}
        ]
    else:
        messages = [
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_prompt}
        ]

    # tokenizing the message object
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )


def process_sorry_bench_sa_data(
    tokenizer: AutoTokenizer,
    random_seed: int = 42,
    evaluation_split: float = 0.2,
) -> DatasetDict:
    '''
        Process the SorryBench SA data into an appropriate format compatible for training
    '''
    
    # loading and shuffling the data
    random.seed(random_seed)
    raw_data = load_sorry_bench_sa_data()
    random.shuffle(raw_data)
    split_idx = int(len(raw_data) * (1 - evaluation_split)) 
    train_data = raw_data[:split_idx]
    eval_data = raw_data[split_idx:]

    # converting to DatasetDict
    dataset_dict = DatasetDict({
        'train': Dataset.from_list(train_data),
        'eval': Dataset.from_list(eval_data)
    })

    # mapping the dataset to a chat template
    def process(example) -> Any:
        return {
            'non_oversight_input': convert_to_model_input(
                example['deployment_framing']['system_prompt'],
                example['deployment_framing']['user_prompt'],
                None,
                tokenizer,
                add_generation_prompt=True
            ),
            'oversight_input': convert_to_model_input(
                example['evaluation_framing']['system_prompt'],
                example['evaluation_framing']['user_prompt'],
                None,
                tokenizer,
                add_generation_prompt=True
            ),
        }
    
    # returning the dataset
    stringified_dataset_dict = dataset_dict.map(process, remove_columns=dataset_dict['train'].column_names)
    return stringified_dataset_dict

def tokenize_dataset(
    processed_dataset: DatasetDict,
    tokenizer: AutoTokenizer
) -> DatasetDict:
    '''
        Tokenizing the processed dataset
    '''
    
    def tokenize_fn(examples):
        non = tokenizer(examples["non_oversight_input"], truncation=True)
        over = tokenizer(examples["oversight_input"], truncation=True)
        return {
            "non_oversight_input_ids": non["input_ids"],
            "non_oversight_attention_mask": non["attention_mask"],
            "oversight_input_ids": over["input_ids"],
            "oversight_attention_mask": over["attention_mask"],
        }
    
    tokenized_dataset_dict = processed_dataset.map(tokenize_fn, batched=True, remove_columns=processed_dataset['train'].column_names)
    return tokenized_dataset_dict

def dataset_to_dataloader(
    tokenized_dataset: DatasetDict,
    tokenizer: AutoTokenizer,
    batch_size: int
) -> DataLoader:
    '''
        Converting the tokenized dataset into a DataLoader
    '''

    # Custom collate_fn to pad each of our four fields separately
    def collate_fn(examples):
        non_batch = tokenizer.pad(
            {
                "input_ids": [e["non_oversight_input_ids"] for e in examples],
                "attention_mask": [e["non_oversight_attention_mask"] for e in examples],
            },
            padding=True,
            return_tensors="pt",
        )

        over_batch = tokenizer.pad(
            {
                "input_ids": [e["oversight_input_ids"] for e in examples],
                "attention_mask": [e["oversight_attention_mask"] for e in examples],
            },
            padding=True,
            return_tensors="pt",
        )

        return {
            "non_oversight_input_ids": non_batch["input_ids"],
            "non_oversight_attention_mask": non_batch["attention_mask"],
            "oversight_input_ids": over_batch["input_ids"],
            "oversight_attention_mask": over_batch["attention_mask"],
        }
    train_dataloader = DataLoader(
        tokenized_dataset['train'],
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    eval_dataloader = DataLoader(
        tokenized_dataset['eval'],
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    return train_dataloader, eval_dataloader


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained('allenai/OLMo-2-1124-7B-Instruct')
    dataset_dict = process_sorry_bench_sa_data(tokenizer=tokenizer)
    dataset_dict = tokenize_dataset(dataset_dict, tokenizer=tokenizer)
    train_dataloader, eval_dataloader = dataset_to_dataloader(
        dataset_dict,
        tokenizer=tokenizer,
        batch_size=2
    )




