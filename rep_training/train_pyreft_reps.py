import argparse
import json
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModel, AutoTokenizer


class TextDataset(Dataset):
    """Very small text dataset that reads one example per line."""

    def __init__(self, texts: List[str]) -> None:
        self.texts = texts

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> str:
        return self.texts[idx]


def load_target_vector(path: Path) -> torch.Tensor:
    """Load a target vector from .npy or .json and return a 1D tensor."""

    if path.suffix == ".npy":
        vec = np.load(path)
    elif path.suffix == ".json":
        with path.open("r") as f:
            vec = json.load(f)
    else:
        raise ValueError(f"Unsupported target vector format: {path.suffix}")

    arr = np.asarray(vec, dtype="float32")
    if arr.ndim != 1:
        raise ValueError("Target vector must be 1D")
    return torch.from_numpy(arr)


def apply_pyreft(model: torch.nn.Module, layer_index: int) -> torch.nn.Module:
    """Wrap the base model with PyReFT modules targeting a given layer.

    This is a minimal placeholder. Integrate your actual PyReFT setup here,
    for example by constructing a PyReFT config and wrapping the model.

    The training script below assumes that all trainable parameters belong to
    these PyReFT modules and can be identified via name substrings.
    """

    try:
        import pyreft  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError("pyreft must be installed to use this script") from exc

    # Replace this no-op with real PyReFT wrapping, for example:
    #   from pyreft import ReftConfig, ReftModel
    #   reft_config = ReftConfig(target_layer=layer_index, ...)
    #   model = ReftModel.wrap(model, reft_config)
    return model


def select_reft_parameters(
    model: torch.nn.Module,
    name_substrings: Iterable[str],
) -> List[torch.nn.Parameter]:
    """Freeze all parameters except those whose names contain given substrings."""

    substrings = list(name_substrings)
    trainable: List[torch.nn.Parameter] = []
    for name, param in model.named_parameters():
        if any(s in name for s in substrings):
            param.requires_grad = True
            trainable.append(param)
        else:
            param.requires_grad = False
    if not trainable:
        joined = ", ".join(substrings)
        raise ValueError(
            f"No parameters matched PyReFT substrings {joined}. "
            "Check your PyReFT integration and naming."
        )
    return trainable


def get_layer_representation(
    hidden_states: Tuple[torch.Tensor, ...],
    layer_index: int,
    attention_mask: torch.Tensor,
    pool: str = "mean",
) -> torch.Tensor:
    """Extract and pool representations from a specific hidden layer.

    hidden_states: tuple of tensors (layers, batch, seq_len, dim)
    layer_index: which layer to use (supports negative indexing)
    pool: "mean" over unmasked tokens or "last" token.
    """

    layer_hidden = hidden_states[layer_index]
    if pool == "last":
        lengths = attention_mask.sum(dim=-1) - 1
        return layer_hidden[torch.arange(layer_hidden.size(0)), lengths]
    if pool == "mean":
        mask = attention_mask.unsqueeze(-1)
        summed = (layer_hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp_min(1)
        return summed / denom
    raise ValueError(f"Unknown pooling method: {pool}")


def collate_and_tokenize(
    batch: List[str],
    tokenizer: AutoTokenizer,
    max_length: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    encoded = tokenizer(
        batch,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    return encoded["input_ids"], encoded["attention_mask"]


def train(
    model_name_or_path: str,
    target_vector_path: Path,
    layer_index: int,
    device: str = "cuda",
    train_file: Path | None = None,
    batch_size: int = 4,
    lr: float = 1e-4,
    max_steps: int = 100,
    max_length: int = 256,
    reft_name_substrings: Iterable[str] = ("pyreft", "reft"),
    pool: str = "mean",
    output_dir: Path | None = None,
) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(
        model_name_or_path,
        output_hidden_states=True,
    )
    model = apply_pyreft(model, layer_index)

    target_vec = load_target_vector(target_vector_path).to(device)

    if train_file is None:
        texts = ["Example input for representation training."]
    else:
        with train_file.open("r") as f:
            texts = [line.strip() for line in f if line.strip()]
        if not texts:
            raise ValueError("Training file is empty")

    dataset = TextDataset(texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model.to(device)
    model.train()

    trainable_params = select_reft_parameters(model, reft_name_substrings)
    optimizer = torch.optim.AdamW(trainable_params, lr=lr)
    loss_fn = torch.nn.MSELoss()

    step = 0
    while step < max_steps:
        for texts_batch in dataloader:
            input_ids, attention_mask = collate_and_tokenize(
                texts_batch,
                tokenizer,
                max_length=max_length,
            )
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            reps = get_layer_representation(
                outputs.hidden_states,
                layer_index=layer_index,
                attention_mask=attention_mask,
                pool=pool,
            )

            if reps.size(-1) != target_vec.size(0):
                raise ValueError(
                    f"Representation dim {reps.size(-1)} does not match "
                    f"target dim {target_vec.size(0)}"
                )

            target = target_vec.unsqueeze(0).expand_as(reps)
            loss = loss_fn(reps, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            step += 1
            if step % 10 == 0:
                print(f"Step {step}/{max_steps} - loss={loss.item():.6f}")
            if step >= max_steps:
                break

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        # Save only parameters that belong to PyReFT modules by name.
        state_dict = model.state_dict()
        filtered = {
            name: tensor
            for name, tensor in state_dict.items()
            if any(s in name for s in reft_name_substrings)
        }
        if not filtered:
            print("Warning: no PyReFT parameters found to save.")
        torch.save(filtered, output_dir / "pyreft_reps.pt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train PyReFT parameters so that a chosen layer's "
            "representation matches a target vector."
        )
    )
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--target_vector_path", type=str, required=True)
    parser.add_argument("--layer_index", type=int, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--train_file", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument(
        "--reft_name_substrings",
        type=str,
        nargs="+",
        default=["pyreft", "reft"],
        help=(
            "Substrings used to identify PyReFT parameters by name; "
            "only these will be trainable and saved."
        ),
    )
    parser.add_argument(
        "--pool",
        type=str,
        choices=["mean", "last"],
        default="mean",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir is not None else None
    train_file = Path(args.train_file) if args.train_file is not None else None
    train(
        model_name_or_path=args.model_name_or_path,
        target_vector_path=Path(args.target_vector_path),
        layer_index=args.layer_index,
        device=args.device,
        train_file=train_file,
        batch_size=args.batch_size,
        lr=args.lr,
        max_steps=args.max_steps,
        max_length=args.max_length,
        reft_name_substrings=args.reft_name_substrings,
        pool=args.pool,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
