import pandas as pd
from datasets import Dataset
from trl import DPOTrainer, DPOConfig
import os
from dataclasses import dataclass, field
from typing import Optional, List
from huggingface_hub import login
from transformers import AutoTokenizer
from transformers import HfArgumentParser
import transformers
from models.instruction_tuned_model import *
from models.qlora_model import load_4bit_model_for_inference
from argparse import Namespace
import fire
import joblib
















import pandas as pd
import transformers
from datasets import Dataset
from typing import Dict, List, Optional

META_PROMPT = (
    "Below is an instruction that describes a task, paired with an input that "
    "provides further context. Write a response that appropriately completes "
    "the request.\n\n"
    "### Instruction:\n{Input}\n\n"
    "### Response:"
)

BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}


def format_prompt(
    example: Dict[str, str],
    meta_prompt: str = META_PROMPT,
) -> str:
    """
    Build the prompt string that will be stored in the `prompt` column.
    Output text is intentionally excluded — TRL handles chosen/rejected
    as separate fields.
    """
    prompt_format = (
        BASE_PROMPT_DICT["prompt_input"]
        if (example.get("input") or "").strip()
        else BASE_PROMPT_DICT["prompt_no_input"]
    )
    formatted_input = prompt_format.format(**example)
    return meta_prompt.format(Input=formatted_input)


def convert_csv_to_dpo_format(
    csv_path: str,
    tokenizer: transformers.PreTrainedTokenizer,
    max_prompt_length: int = 256,
    max_response_length: int = 256,
) -> Dataset:

    df = pd.read_csv(csv_path).fillna("")

    if "input" not in df.columns:
        df["input"] = ""

    df = df[df["preference"].astype(str).isin(["1", "2"])]
    for col in ("instruction", "output_1", "output_2"):
        df = df[df[col].str.strip() != ""]
    df = df.reset_index(drop=True)

    eos = tokenizer.eos_token or ""
    prompts, chosen_list, rejected_list = [], [], []

    for _, row in df.iterrows():
        example = {"instruction": row["instruction"], "input": row["input"]}

        raw_prompt = format_prompt(example)

        prompt_ids = tokenizer(
            raw_prompt,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        )["input_ids"]
        prompt_text = tokenizer.decode(prompt_ids, skip_special_tokens=True)

        if int(row["preference"]) == 1:
            chosen_raw, rejected_raw = row["output_1"], row["output_2"]
        else:
            chosen_raw, rejected_raw = row["output_2"], row["output_1"]

        def encode_response(text: str) -> str:
            """Truncate then re-attach EOS so it survives truncation."""
            ids = tokenizer(
                text,
                truncation=True,
                max_length=max_response_length,
                add_special_tokens=False,
            )["input_ids"]
            # Decode without special tokens, then append EOS manually
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
            return decoded + eos

        prompts.append(prompt_text)
        chosen_list.append(encode_response(chosen_raw))
        rejected_list.append(encode_response(rejected_raw))

    dataset = Dataset.from_dict(
        {"prompt": prompts, "chosen": chosen_list, "rejected": rejected_list}
    )

    print("Sample:\n", dataset[0])
    return dataset












def main(
    preference_data: str = "...",
    batchSize: int = 1,
    outDir: str = "DPO/iteration1_reference_base",
    modelString: str = "meta-llama/Llama-3.2-3B",
    hf_token: str = None  # Add parameter for token
):
    
    token = hf_token or os.getenv("HF_TOKEN")
    if token:
        login(token=token)
    else:
        print("Warning: No HuggingFace token provided. Set HF_TOKEN environment variable or pass hf_token parameter.")

    
    training_args_dpo = DPOConfig(
        model_adapter_name="current_policy",
        ref_adapter_name="reference_policy",  


        
        output_dir=outDir,
    
        max_length=512,
        max_prompt_length=256,                 
    
        per_device_train_batch_size=batchSize,
        gradient_accumulation_steps=4,   
    
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},  # required for QLoRA + PEFT
    
        bf16=True,
        optim="paged_adamw_32bit",            
        learning_rate=5e-5,                  
        lr_scheduler_type="cosine",
        warmup_ratio=0.06,
    
        dataset_num_proc=4,                  
    
        num_train_epochs=1.0,
            
        label_smoothing = 0.1,
        logging_steps=500,
        save_steps=1000,
        save_total_limit=10,
        resume_from_checkpoint=None,
        eval_strategy="no",
    )
    tokenizer = AutoTokenizer.from_pretrained(
        modelString,
        cache_dir=None,
        model_max_length=training_args_dpo.max_length,
        padding_side="right",
    )
    
    config = InstructionConfig(backbone_model_name_or_path=modelString)

    @dataclass
    class ModelArguments:
        model_name_or_path: Optional[str] = field(default=modelString)
    
    @dataclass
    class TrainingArguments(transformers.TrainingArguments):
        cache_dir: Optional[str] = field(default=None)
        optim: str = field(default="paged_adamw_32bit")
        model_max_length: int = field(
            default=512,
            metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
        )
        full_finetune: bool = field(default=False)
        adam8bit: bool = field(default=False)
        double_quant: bool = field(default=True)
        quant_type: str = field(default="nf4")
        bits: int = field(default=4)
        lora_modules: Optional[List[str]] = field(default=None)
        lora_r: int = field(default=64)
        lora_alpha: int = field(default=16)
        lora_dropout: float = field(default=0.0)
        remove_unused_columns: bool = field(default=False)
        resume_from_training: bool = field(default=True)
        fp16: bool = field(default=True)
        bf16: bool = field(default=False)
        batch_eval_metrics: Optional[str] = field(default=None)
        eval_strategy: str = field(default="no")
        trust_remote_code: bool = field(default=True)



        
    parser = transformers.HfArgumentParser((ModelArguments, TrainingArguments));
    
    model_args, training_args = parser.parse_args_into_dataclasses(args=[]);

    args = Namespace(**vars(model_args), **vars(training_args))
    
    
    reference_policy = InstructionModel(
        args=args,
        config=config,
        qlora=True,
        adapter_name = "reference_policy",
    )
    

    # reference_policy = load_4bit_model_for_inference(
    #     checkpoint_dir="DPO/iteration0/checkpoint-3943/current_policy",
    #     bits=4,
    #     fp16=args.fp16,
    #     bf16=args.bf16,
    #     gradient_checkpointing=True,   # saves memory
    #     adapter_name="reference_policy",
    #     is_trainable=False,
    #     reuse_base_model=False,
    #     trust_remote_code=True,
    #     base_model_mapping=None,
    #     fully_initialize=False,        # << do NOT fully init again
    #     base_model_name_or_path_for_fully_initialize=modelString,
    # )
    
    current_policy = load_4bit_model_for_inference(
        checkpoint_dir="DPO/iteration0/checkpoint-3943/current_policy",
        bits=4,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=True,   # saves memory
        adapter_name="current_policy",
        is_trainable=True,
        reuse_base_model=False,
        trust_remote_code=True,
        base_model_mapping=None,
        fully_initialize=False,        # << do NOT fully init again
        base_model_name_or_path_for_fully_initialize=modelString,
    )



    
    import inspect
    
    # list all methods of current_policy
    methods = [m[0] for m in inspect.getmembers(current_policy, predicate=inspect.ismethod)]
    print(methods)

    

    if tokenizer.pad_token is None:
        tokenizer.pad_token_id = (
            0  
        )


    dpo_dataset = convert_csv_to_dpo_format(preference_data, tokenizer = tokenizer)

    print("Sample from converted dataset:")
    print(dpo_dataset[0])
    
    print(f"Saving dataset to disk...")
    dpo_dataset.save_to_disk('data/dpo_dataset')
    
    print(f"Initializing DPO Trainer...")
    trainer = DPOTrainer(
        model=current_policy,
        ref_model = reference_policy,
        args=training_args_dpo,
        train_dataset=dpo_dataset,
        processing_class=tokenizer
    )

    print(f"Starting DPO training...")
    trainer.train()
    
    print(f"Saving final model to {outDir}")
    trainer.save_model(outDir)
    tokenizer.save_pretrained(outDir)
    
    print("Training completed!")


if __name__ == "__main__":
    fire.Fire(main)