from dataclasses import dataclass, field
from utils.vllm_utils import model_dictionary

@dataclass
class ModelArguments:
    model_name: str = field(
        default='olmo2-7b-instruct',
        metadata={
            'help': 'Model name for training',
            'choices': list(model_dictionary.keys())
        }
    )
    output_dir: str = field(
        default='training_output/olmo2-7b-instruct/activation_mapper/test',
        metadata={'help': 'Output directory for saving the trained activation mapper'},
    )

@dataclass
class LoraArguments:
    lora_r: int = field(
        default=16,
        metadata={'help': 'LoRA rank for training the activation mapper'},
    )
    layers_to_transform: int = field(
        default=15,
        metadata={'help': 'The layer to transform for activation mapping'},
    )
    lora_alpha: int = field(
        default=32,
        metadata={'help': 'LoRA alpha for training the activation mapper'},
    )
    lora_dropout: float = field(
        default=0.05,
        metadata={'help': 'LoRA dropout for training the activation mapper'},
    )
    target_modules: str = field(
        default='mlp',
        metadata={
            'help': 'Target modules for LoRA training the activation mapper',
            'choices': ['all', 'mlp', 'qkv']
        },
    )

@dataclass
class TrainingArguments:
    num_epochs: int = field(
        default=10,
        metadata={'help': 'Number of epochs for training the activation mapper'},
    )
    batch_size: int = field(
        default=4,
        metadata={'help': 'Batch size for training the activation mapper'},
    )
    gradient_accumulation_steps: int = field(
        default=2,
        metadata={'help': 'Gradient accumulation steps for training the activation mapper'},
    )
    learning_rate: float = field(
        default=9e-4,
        metadata={'help': 'Learning rate for training the activation mapper'},
    )
    weight_decay: float = field(
        default=0.01,
        metadata={'help': 'Weight decay for training the activation mapper'},
    )
    warmup_ratio: float = field(
        default=0.06,
        metadata={'help': 'Warmup ratio for training the activation mapper'},
    )
    plot: bool = field(
        default=False,
        metadata={'help': 'Whether to plot UMAP of representations during training'}
    )

@dataclass
class ActivationMapperArguments:
    approach: str = field(
        default='consistency',
        metadata={
            'help': 'Activation mapping approach to use',
            'choices': ['consistency', 'redirection']
        }
    )