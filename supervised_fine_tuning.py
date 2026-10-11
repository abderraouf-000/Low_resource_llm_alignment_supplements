import copy
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List

import torch
import transformers
from torch.utils.data import Dataset
from transformers import Trainer
from argparse import Namespace
import transformers
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM


from models.instruction_tuned_model import *
from data_utils.instruction_fine_tuning import *

IGNORE_INDEX = -100
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"

from huggingface_hub import login


HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN);

torch.backends.cuda.matmul.allow_tf32 = True

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

modelArgs = Namespace(
device = "cuda" if torch.cuda.is_available() else "cpu",
model_name_or_path = "meta-llama/Llama-3.2-3b",
);


tokenizer = transformers.AutoTokenizer.from_pretrained(
    modelArgs.model_name_or_path,
    cache_dir=None,
    model_max_length= 512,
    padding_side="right",
    use_fast=False,
)




@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="meta-llama/Llama-3.2-3b")


@dataclass
class DataArguments:
    data_path: str = field(default="data/alpaca_data.json", metadata={"help": "Path to the training data."})


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="paged_adamw_32bit")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    padding: str = field(default="longest");
    end_sequence_with_eos: bool = field(default=False)
    full_finetune: bool = field(default=False)
    adam8bit: bool = field(default=False)
    double_quant: bool = field(default=True)
    quant_type: str = field(default="nf4")
    bits: int = field(default = 4)
    lora_modules: Optional[List[str]] = field(default=None)
    lora_r: int = field(default=32)
    lora_alpha: int = field(default=16)
    lora_dropout: float = field(default=0.0)
    report_to: str = field(default="none")
    resume_dir: Optional[str] = field(default=None)
    per_device_train_batch_size: int = field(default=4)
    gradient_accumulation_steps: int = field(default=2)
    weight_decay: float = field(default=0.0)
    learning_rate: float = field(default=5e-5)
    remove_unused_columns: bool = field(default=False)
    max_grad_norm: float = field(default=0.3)
    gradient_checkpointing: bool = field(default=True)
    do_train: bool = field(default=True)
    lr_scheduler_type: str = field(default="constant")
    warmup_ratio: float = field(default=0.03)
    logging_steps: int = field(default=3000)
    group_by_length: bool = field(default=False)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=5000)
    save_total_limit: int = field(default=20)
    resume_from_training: bool = field(default=True)
    fp16: bool = field(default=True)
    bf16: bool = field(default=False)
    batch_eval_metrics: Optional[str] = field(default=None)
    eval_strategy: str = field(default="no")
    trust_remote_code: bool = field(default = True)
    output_dir:str = field(default = "Instruction_following_alpaca_3B_lora_32")


parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args = [])


args = Namespace(**vars(model_args),**vars(training_args),**vars(data_args));

config = InstructionConfig(backbone_model_name_or_path=model_args.model_name_or_path)

model = InstructionModel(
    args=args,
    config=config,
    qlora=True,
    adapter_name = "lora_policy",
    # checkpoint_dir = "trainer_output/checkpoint-",
)


model.backbone_model.config.use_cache = False;


if tokenizer.pad_token is None:
    tokenizer.pad_token_id = (
        0  
    )

    

    

data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args, skipped_steps = 0, add_eos_token = True);


trainer = Trainer(model=model, processing_class=tokenizer, args=training_args, **data_module)


trainer.train()




