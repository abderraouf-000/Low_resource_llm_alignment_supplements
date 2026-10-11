import argparse
from dataclasses import dataclass, field
import os
import sys
from os.path import exists, join, isdir
import shutil
from typing import Optional, List
import logging
from argparse import Namespace
import torch
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from data_utils.data_utils_ppo import make_rl_data_module
from models.ppo_trainer_base import PPOTrainer, make_models
from transformers import HfArgumentParser
from huggingface_hub import login


torch.backends.cuda.matmul.allow_tf32 = True

DEFAULT_PAD_TOKEN = "[PAD]"

HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)


@dataclass
class ModelArguments:
    reward_model_name_or_path: str = ""
    second_reward_model_name_or_path: str = ""
    base_model_name_or_path_for_fully_initialize: str = "meta-llama/Llama-3.2-3b"
    model_directory: str = "meta-llama/Llama-3.2-3b"


@dataclass
class DataArguments:
    policy_meta_prompt_pattern: str = "prompts/policy_meta_prompt_pattern.txt"
    reward_meta_prompt_pattern: str = "prompts/sft_reward_prompt-2.txt"
    dataset_path: str = ""
    eval_dataset_path: str = ""
    train_splits: List[str] = field(default_factory=lambda: ["train"])
    eval_splits: List[str] = field(default_factory=lambda: ["train"])
    principle_collection_path: str = "prompts/Reward_model_dimensions/principle_collection_ppo.json"
    max_principles: int = 3
    stop_token: Optional[str] = None
    dataset_name: Optional[str] = None


@dataclass
class TrainingArguments(transformers.Seq2SeqTrainingArguments):

    # Core training
    remove_unused_columns: bool = False
    gradient_checkpointing: bool = True
    do_train: bool = True
    learning_rate: float = 1e-5
    lr_scheduler_type: str = "constant"
    warmup_steps: int = 5
    weight_decay: float = 0.0
    max_grad_norm: float = 0.3
    fp16: bool = True
    adam8bit: bool = False
    gradient_accumulation_steps: int = 2
    per_device_train_batch_size: int = 1
    total_epochs: int = 1
    
    # dataloader_drop_last=False,
    # dataloader_num_workers=0,
    group_by_length=False,

    output_dir: str = ""
    save_total_limit: int = 40
    save_strategy: str = "steps"
    save_steps: int = 50
    logging_steps: int = 50
    report_to: str = "none"
    trust_remote_code: bool = True
    full_determinism: bool = False
    full_finetune: bool = False

    resume_from_training: bool = True
    resume_dir: str = ""
    cache_dir: str = None
    skipped_data: int = 0
    verifiable_reward : bool = False
    reference_model_dir: str = ""

    # Evaluation
    eval_strategy: str = "steps"
    do_eval: bool = False
    eval_batches: int = sys.maxsize
    eval_steps: int = 50

    # PPO hyperparameters
    kl_coef: float = 0.03
    target_kl: float = 6.0
    cliprange: float = 0.2
    cliprange_value: float = 0.2
    vf_coef: float = 0.1
    lam: float = 1.0
    gamma: float = 1.0
    noptepochs: int = 2
    k_beta: float = 0.1
    kl_approximator: str = "k1"
    adaptive_kl: bool = False

    # Rollouts & Sampling
    query_len: int = 256
    response_len: int = 256
    model_max_length: int = 512
    temperature: float = 0.7
    step_batch_size: int = 1
    rollout_batch_size: int = 1
    rollout_per_device_batch_size: int = 1
    rollout_accumulation_steps: int = 1
    step_per_device_batch_size: int = 1

    
    # Reward Model & Values
    reward_model_bits: int = 4
    reward_model_per_device_batch_size: int = None
    penalty_reward_value: float = -0.5
    bonus_answer_NOANSWER_questions: float = 0.0
    early_stop_penalty: float = -2.0
    whiten_rewards: bool = False
    whitening_async_stats: str = "full_batch"
    init_value_with_reward: bool = True

    
    # Token Handling
    clean_tokens_after_eos: bool = True
    truncate_tokens: int = None
    truncate_after: int = None

    # Length Bonuses
    length_bonus_score: float = 2.0
    length_bonus_upper_bound: float = 0.3

    # Stop Token Penalties
    penalize_no_stop_token: bool = True
    relative_stop_token_penalty: bool = True

    # Advanced Options
    ddp_find_unused_parameters: bool = False
    enable_redteaming_principles: bool = True
    enable_helpfulness_principles: bool = True

    # Optimization
    optim: str = "paged_adamw_32bit"
    config_file_name: str = None
    policy_model_bits: int = 4
    min_token_limit: int = None
    fully_initialize_policy: bool = False
    base_model_mapping: dict = None
    save_steps_extra: str = None
    load_optimizer:bool = False
    
    # save_steps_extra_list: List[] = field(default_factory=list)
    use_gmm: bool = False


    
def set_truncate_token_ids(tokenizer: transformers.PreTrainedTokenizer, training_args):
    """Convert truncation token to token ids. Called in RLTrainer."""
    truncate_tokens = training_args.truncate_tokens

    if truncate_tokens is None:
        truncate_token_ids = None
    else:
        truncate_token_ids = tokenizer.convert_tokens_to_ids(truncate_tokens)

    training_args.truncate_token_ids = truncate_token_ids


def main():
    parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments));
    
    training_args, model_args, data_args = parser.parse_args_into_dataclasses()

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_directory,
        cache_dir=None,
        model_max_length=training_args.model_max_length,
        padding_side="left",
        truncation_side="right",
    )
    

    set_truncate_token_ids(tokenizer, training_args)

    tokenizer.pad_token_id = 0

    data_module = make_rl_data_module(
        tokenizer=tokenizer, data_args=data_args, training_args=training_args
    )

    args = argparse.Namespace(**vars(model_args), **vars(data_args), **vars(training_args))

    training_args.data_config = data_args


    model_module = make_models(
        tokenizer=tokenizer,
        reward_tokenizer=tokenizer,
        args=args,
        num_new_tokens=0,
        reward_num_new_tokens=0,
        resume_from_checkpoint=(
            training_args.resume_dir if training_args.resume_from_training else None
        ),
    )

    
    if not exists(training_args.output_dir):
        os.makedirs(training_args.output_dir)

    
    trainer = PPOTrainer(
        args=training_args,
        **data_module,
        **model_module,
        optimizer=None,
        lr_scheduler=None,
        tokenizer=tokenizer,
    )

    trainer.train(
        resume_training_ckpt=(
            training_args.resume_dir if training_args.resume_from_training else None
        )
    )


if __name__ == "__main__":
    main()