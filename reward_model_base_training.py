import argparse
from argparse import Namespace
from dataclasses import dataclass, field
from datasets import load_dataset, disable_caching
import transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
)
from transformers import PreTrainedModel, AutoConfig
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from transformers.utils.generic import ModelOutput
from transformers.trainer_utils import EvalPrediction
import torch
import tqdm
import json
import random
import pandas as pd
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
from transformers import AutoModel
import os
from typing import Optional, Dict, Sequence, List, Tuple
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence
import einops
import bitsandbytes as bnb
from peft import PeftModelForCausalLM
from peft.tuners.lora import LoraLayer
from transformers import LlamaForCausalLM

from models.reward_model import (
    RewardConfig,
    RewardModel,
    RewardModelTrainer,
    compute_reward_modeling_metrics,
)

disable_caching()

torch.backends.cuda.matmul.allow_tf32 = True

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

REGISTERED_BASE_MODELS = {}

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}


sft_reward_model_prompt = """
You are a reviewer whose goal is to judge the quality of the AI system's responses to instructions.

Your task is to evaluate the quality of the response. There are several dimensions you should consider in your evaluation:

- The AI should be tailored to the nature of the user query, taking into account who is interacting with the AI, as well as the situational context in which the assistant is being engaged.

A good response should meet all of the above criteria.


User: {Input}

response: {Output}

The quality of the response is
"""

@dataclass
class ModelArguments:
    model_name_or_path: str = "meta-llama/Llama-3.2-3b"
    trust_remote_code: bool = False
    checkpoint_dir: str = "base_reward_model/checkpoint400_iteration2_reward_model_alpaca_grounded_prose"
    adapter_name: str = "lora_default"


@dataclass
class DataArguments:
    dataset_path: str = ""
    dataset_name: Optional[str] = None
    eval_size: int = 50
    n_samples: Optional[int] = None
    principle_collection_path: Optional[str] = None


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    # Model / quantization
    model_max_length: int = 512
    full_finetune: bool = False
    adam8bit: bool = False
    double_quant: bool = True
    quant_type: str = "nf4"
    bits: int = 4
    lora_modules: Optional[str] = None
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.0

    # Trainer basics
    output_dir: str = ""
    num_train_epochs: int = 1
    per_device_train_batch_size: int = 10
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 5e-5
    weight_decay: float = 0.0
    warmup_ratio: float = 0.03
    lr_scheduler_type: str = "cosine"
    max_grad_norm: float = 0.3
    optim: str = "paged_adamw_32bit"
    fp16: bool = True
    bf16: bool = False

    # Logging / saving
    logging_steps: int = 300
    save_strategy: str = "steps"
    save_steps: int = 1000
    save_total_limit: int = 40
    report_to: str = "none"
    group_by_length: bool = True

    # Eval
    eval_strategy: str = "steps"
    eval_steps: int = 1000

    # Misc
    remove_unused_columns: bool = False
    label_names: Optional[List[str]] = field(default_factory=lambda: ["index_0", "index_1", "choice"])
    gradient_checkpointing: bool = True
    cache_dir: Optional[str] = None
    resume_from_training: bool = True
    resume_dir: Optional[str] = None
    save_safetensors: bool = False
    end_sequence_with_eos: bool = False


def print_trainable_parameters(args, model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    if args.bits == 4:
        trainable_params /= 2
    print(
        f"trainable params: {trainable_params} || "
        f"all params: {all_param} || "
        f"trainable: {100 * trainable_params / all_param}"
    )


def format_full_prompt(
    example: Dict[str, str],
    meta_prompts: List[str],
    tokenizer: transformers.PreTrainedTokenizer,
    eos_token: Optional[str] = None,
    query_len: Optional[int] = None,
    response_len: Optional[int] = None,
    output_key: str = "output",
) -> str:
    if eos_token is None:
        eos_token = ""

    if "example_id" in example:
        total_meta_prompt = len(meta_prompts)
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    else:
        meta_prompt = meta_prompts[0]

    prompt_format = (
        BASE_PROMPT_DICT["prompt_input"]
        if (example.get("input") or "").strip()
        else BASE_PROMPT_DICT["prompt_no_input"]
    )

    formatted_input = prompt_format.format(**example)
    formatted_output = example[output_key]

    formatted_prompt = (
        meta_prompt.format(
            Input=formatted_input,
            Output=formatted_output,
        )
        + eos_token
    )

    return formatted_prompt


def _tokenize_fn(strings, tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""

    def _tokenize(x):
        tokenized_text = tokenizer(
            x["full_prompt"],
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        return {
            "input_ids": tokenized_text.input_ids,
            "attention_mask": tokenized_text.attention_mask,
            "input_length": tokenized_text.attention_mask.sum(),
        }

    tokenized_list = strings.map(lambda x: _tokenize(x))
    tokenized_list.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "input_length"],
    )

    input_ids = labels = [
        tokenized["input_ids"][0]
        for tokenized in tqdm.tqdm(tokenized_list, desc="Concatenating input_ids")
    ]

    input_ids_lens = labels_lens = [
        tokenized["input_length"].item()
        for tokenized in tqdm.tqdm(tokenized_list, desc="Computing lengths")
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(sequences, batch_first, padding_value)
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence


def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng


def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(train_dataset), \
        "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset



class RewardModellingDataSet(Dataset):
    """Dataset for reward model fine-tuning."""

    def __init__(self, data, meta_prompts, tokenizer: transformers.PreTrainedTokenizer):
        super(RewardModellingDataSet, self).__init__()
        list_dict_data = data

        index_0, index_1 = tuple(
            torch.full(size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long)
            for fill_value in (0, 1)
        )

        preference_values = [dict_data["preference"] for dict_data in list_dict_data]
        unique_prefs = set(preference_values)

        if unique_prefs.issubset({0, 1}):
            preference_map = {0: 0, 1: 1}
            print(f"Detected 0-indexed preferences: {unique_prefs}")
        elif unique_prefs.issubset({1, 2}):
            preference_map = {1: 0, 2: 1}
            print(f"Detected 1-indexed preferences: {unique_prefs}")
        else:
            print(f"Warning: Unexpected preference values found: {unique_prefs}")
            sorted_prefs = sorted(unique_prefs)
            preference_map = {val: idx for idx, val in enumerate(sorted_prefs)}
            print(f"Using mapping: {preference_map}")

        choice = torch.tensor(
            [[preference_map[dict_data["preference"]]] for dict_data in list_dict_data]
        )

        if "preference_helpful" in list_dict_data[0].keys():
            choice_helpful = torch.tensor(
                [[preference_map[dict_data["preference_helpful"]]] for dict_data in list_dict_data]
            )
        else:
            choice_helpful = None

        if "is_output_1_safe" in list_dict_data[0].keys():
            is_output_1_safe = torch.tensor(
                [[dict_data.get("is_output_1_safe", True)] for dict_data in list_dict_data],
                dtype=torch.bool,
            )
            is_output_2_safe = torch.tensor(
                [[dict_data.get("is_output_2_safe", True)] for dict_data in list_dict_data],
                dtype=torch.bool,
            )
        else:
            is_output_1_safe = None
            is_output_2_safe = None

        text_list_0 = list_dict_data.map(
            lambda example: {"full_prompt": format_full_prompt(
                example, meta_prompts, tokenizer, tokenizer.eos_token, None, None, "output_1",
            )}
        )

        text_list_1 = list_dict_data.map(
            lambda example: {"full_prompt": format_full_prompt(
                example, meta_prompts, tokenizer, tokenizer.eos_token, None, None, "output_2",
            )}
        )

        tokenized_0, tokenized_1 = tuple(
            _tokenize_fn(text_list, tokenizer)
            for text_list in (text_list_0, text_list_1)
        )

        input_ids = [list(pair) for pair in zip(tokenized_0["input_ids"], tokenized_1["input_ids"])]
        labels = [list(pair) for pair in zip(tokenized_0["labels"], tokenized_1["labels"])]

        self.input_ids = input_ids
        self.labels = labels
        self.index_0 = index_0
        self.index_1 = index_1
        self.choice = choice
        self.is_output_1_safe = is_output_1_safe
        self.is_output_2_safe = is_output_2_safe
        self.choice_helpful = choice_helpful


    def __len__(self):
        return len(self.input_ids)


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            index_0=self.index_0[i],
            index_1=self.index_1[i],
            choice=self.choice[i],
            is_output_1_safe=self.is_output_1_safe[i] if self.is_output_1_safe is not None else None,
            is_output_2_safe=self.is_output_2_safe[i] if self.is_output_2_safe is not None else None,
            choice_helpful=self.choice_helpful[i] if self.choice_helpful is not None else None,
        )


class DataCollatorForRewardDataset:
    """Collate examples for reward model training."""

    def __init__(self, tokenizer: transformers.PreTrainedTokenizer):
        self.tokenizer = tokenizer

    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        input_ids = [seq for instance in instances for seq in instance[key]]
        input_ids = pad_sequence_from_left(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id,
        )
        input_ids = einops.rearrange(
            input_ids,
            "(bsz num_candidates) max_seq_len -> bsz num_candidates max_seq_len",
            num_candidates=len(instances[0][key]),
        )
        return input_ids

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        index_0, index_1, choice = tuple(
            torch.stack([instance[key] for instance in instances])
            for key in ("index_0", "index_1", "choice")
        )
        input_ids = self._left_pad_helper(instances, "input_ids")
        attention_mask = input_ids.ne(self.tokenizer.pad_token_id).long()

        return dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        )
    








def make_binary_reward_modeling_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
):
    meta_prompts = [sft_reward_model_prompt]
    
    print(data_args)
    
    if data_args.dataset_path.endswith("json"):
        train_preference = load_dataset("json", data_files=data_args.dataset_path)["train"]
        if data_args.n_samples:
            train_preference = train_preference.select(range(data_args.n_samples))

    elif data_args.dataset_path.endswith("csv"):
        train_preference = load_dataset("csv", data_files=data_args.dataset_path)["train"]
        if data_args.n_samples:
            train_preference = train_preference.select(range(data_args.n_samples))
    else:
        raise ValueError(f"Unsupported dataset format: {data_args.dataset_path}")

    train_dataset = RewardModellingDataSet(
        data=train_preference,
        meta_prompts=meta_prompts,
        tokenizer=tokenizer,
    )

    train_dataset, eval_dataset = split_train_into_train_and_eval(
        train_dataset=train_dataset,
        eval_size=data_args.eval_size,
        seed=200,
    )

    data_collator = DataCollatorForRewardDataset(tokenizer=tokenizer)

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )



def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments))
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = 0

    data_module = make_binary_reward_modeling_data_module(tokenizer, data_args, training_args)

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    config = RewardConfig(backbone_model_name_or_path=model_args.model_name_or_path)

    model = RewardModel(
        args=args,
        config=config,
        qlora=True,
        checkpoint_dir=model_args.checkpoint_dir if model_args.checkpoint_dir!="" else None,
        adapter_name=model_args.adapter_name,
    )

    model.backbone_model.config.use_cache = False

    print("Loaded model")

    trainer = RewardModelTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        compute_metrics=compute_reward_modeling_metrics,
        **{k: v for k, v in data_module.items() if k != "predict_dataset"},
    )

    results = trainer.train();


    model.save_pretrained(training_args.output_dir)
    print(f"Model saved to {training_args.output_dir}")


if __name__ == "__main__":
    main()