from transformers import AutoTokenizer, AutoModelForCausalLM,PreTrainedTokenizer
from argparse import Namespace
from datasets import load_dataset
from transformers import PreTrainedModel, AutoConfig
import torch.nn as nn
from torch import Tensor
from torch.utils.data import Dataset
from transformers.utils.generic import ModelOutput
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput
import torch
import tqdm
import json
import random
import pandas as pd
from transformers import AutoModel
import fire
import os
from typing import Optional, Dict, Sequence, List
from pathlib import Path
import math
from torch.nn.utils.rnn import pad_sequence
import einops
from typing import Tuple

import sys
import os


BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}

def format_full_prompt(
    example: Dict[str, str], ## done
    meta_prompts: List[str], ## done
    tokenizer: PreTrainedTokenizer, ## done
    eos_token: Optional[str] = None,## done
    query_len: Optional[int] = None, ## doen
    response_len: Optional[int] = None, ## done
    output_key: str = "output", ## done
) -> str:

    
    ## current : Not none
    if eos_token is None:
        eos_token = "";

        

    # if "example_id" in example:
    #     total_meta_prompt = len(meta_prompts);
        
    #     meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    
    # else:
    
    meta_prompt = meta_prompts[0];

        
    # if example.get("input", "") != "":
    #     prompt_format = BASE_PROMPT_DICT["prompt_input"];

        
    # else:
    
    prompt_format = BASE_PROMPT_DICT["prompt_no_input"];

    
    
    formatted_input = prompt_format.format(**example)

    

    # print(f"---------------formatted input{formatted_input}----------------");


    formatted_output = example[output_key]

    formatted_prompt = (
        meta_prompt.format(
            Input=formatted_input,
            Output=formatted_output,
        )
        + eos_token
    )

    print("====================== formatted_prompt ============================",formatted_prompt)

 
    return formatted_prompt







def _tokenize_fn(strings: Sequence[str], tokenizer: PreTrainedTokenizer) -> Dict:

    """Tokenize a list of strings."""


    def _tokenize(x):

        # print("text to toknize : ",x["full_prompt"][0]);
        
        tokenized_text = tokenizer(
            x["full_prompt"],
            return_tensors="pt",
            padding='longest',
            max_length=tokenizer.model_max_length,
            truncation=True,
        )

        return {
            "input_ids": tokenized_text.input_ids,
            "attention_mask": tokenized_text.attention_mask,
            "input_length": tokenized_text.attention_mask.sum(),
        }


    


    tokenized_list = strings.map(
    lambda x: _tokenize(x),
    );



    tokenized_list.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "input_length"],
    )


    input_ids = labels = [
            tokenized["input_ids"][0]
            for tokenized in tqdm.tqdm(
                tokenized_list,
                desc="Concatenating input_ids",
            )]

    input_ids_lens = labels_lens = [
    tokenized["input_length"].item()
    for tokenized in tqdm.tqdm(tokenized_list, desc="Computing lengths")
    ]


    

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    );



    

class RewardModellingDataSet(Dataset):
    """Dataset for supervised fine-tuning."""

    # def __init__(self, data ,  meta_prompts,tokenizer: transformers.PreTrainedTokenizer,principle_collection):
    def __init__(self, data ,  meta_prompts,tokenizer: PreTrainedTokenizer):

        super(RewardModellingDataSet, self).__init__()

        list_dict_data = data;


        ######## Preprocess_for_reward_model #######

        ### create index tensors ###


        index_0, index_1 = tuple(
          torch.full(
        size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long
                )
               for fill_value in (0, 1)
        );


        ### create choice tensor ###

        choice = torch.tensor(
            [[{1: 0, 2: 1}[dict_data["preference"]]] for dict_data in list_dict_data]
        )

        ### Construct prompts for responses ###
        print(format_full_prompt(
            list_dict_data[0], ## done
            # principle_collection,
            meta_prompts, ## done
            tokenizer, ## done
            tokenizer.eos_token,## done (changed)
            None, ## doen
            None, ## done
            "output_1", ## done
            )
        )
        text_list_0 = list_dict_data.map(
            lambda example: {"full_prompt": format_full_prompt(
            example, ## done
            # principle_collection,
            meta_prompts, ## done
            tokenizer, ## done
            tokenizer.eos_token,## done (changed)
            None, ## doen
            None, ## done
            "output_1", ## done
            )}
        )

        text_list_1 = list_dict_data.map(
            lambda example: {"full_prompt": format_full_prompt(
            example, ## done
            # principle_collection,
            meta_prompts, ## done
            tokenizer, ## done
            tokenizer.eos_token,## done (changed)
            None, ## done
            None, ## done
            "output_2", ## done
            )}
        );
        

        #### tokenize examples ####

        tokenized_0, tokenized_1 = tuple(
        _tokenize_fn(text_list, tokenizer)
        for text_list in (text_list_0, text_list_1)
        )

        # print("---- Tokenized sentence : ----\n\n",tokenized_0.keys());

        input_ids = [
        list(pair)
        for pair in zip(tokenized_0["input_ids"], tokenized_1["input_ids"])
        ]

        # print(f"-------------- The input ids 1 are : {input_ids[0][1]}");
        # print(f"-------------- The input ids 2 are : {input_ids[0][0]}");


        labels = [
        list(pair) for pair in zip(tokenized_0["labels"], tokenized_1["labels"])
        ];

        

        packaged_data = dict(
            input_ids=input_ids,
            labels=labels,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        );


        self.input_ids = packaged_data["input_ids"];

        
        print("input ids",self.input_ids[0][0]);
        tokens = tokenizer.convert_ids_to_tokens(self.input_ids[0][0])
        print("Tokens:", tokens)

        
        print("---- decoded sentence : ----\n\n", tokenizer.batch_decode(
                self.input_ids[0],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,))

        
        self.labels = packaged_data["labels"];
        self.index_0 = packaged_data["index_0"];
        self.index_1 = packaged_data["index_1"];
        self.choice = packaged_data["choice"];

        # print(f"input_ids {self.input_ids}");
        # print(f"labels {self.input_ids}");


    def __len__(self):
        # print(f"__len__ : {len(self.input_ids)}")
        return len(self.input_ids)

    # def naive__getitem__(self, i) -> Dict[str, torch.Tensor]:
    #     return dict(input_ids=self.input_ids[i], labels=self.labels[i])

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        # print(self.input_ids[i]);
        # print(
        return  dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i],
            index_0=self.index_0[i],
            index_1=self.index_1[i],
            choice=self.choice[i],
        );





def pad_sequence_from_left(
    sequences: Sequence[torch.Tensor],
    batch_first: bool = False,
    padding_value: float = 0.0,
):
    """Mirror of `torch.nn.utils.rnn.pad_sequence`, but pad from left."""
    sequences = tuple(sequence.flip(0) for sequence in sequences)
    padded_sequence = torch._C._nn.pad_sequence(
        sequences, batch_first, padding_value
    )  # noqa
    padded_sequence = padded_sequence.flip(int(batch_first))
    return padded_sequence;







    

class DataCollatorForRewardDataset():

    """Collate examples for reward model database."""


    def __init__(self,tokenizer : PreTrainedTokenizer):

        self.tokenizer = tokenizer;


    def _left_pad_helper(self, instances: Sequence[dict], key: str):
        input_ids = [seq for instance in instances for seq in instance[key]]  # Flatten.
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


    def __call__(self,instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:

        # print(f"From collator {instances[0].keys()}");


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





def _get_generator(seed: int) -> torch.Generator:
    rng = torch.Generator()
    rng.manual_seed(seed)
    return rng





    
def split_train_into_train_and_eval(
    train_dataset: Dataset, eval_size: int, seed: int
) -> Tuple[Dataset, Dataset]:
    assert eval_size < len(
        train_dataset  # noqa
    ), "Requested eval_size cannot be equal/larger than original train data size."
    new_train_size = len(train_dataset) - eval_size  # noqa
    train_dataset, eval_dataset = torch.utils.data.random_split(
        train_dataset, [new_train_size, eval_size], generator=_get_generator(seed)
    )
    return train_dataset, eval_dataset



    


    
def make_binary_reward_modeling_data_module(
    tokenizer: PreTrainedTokenizer,
    data_args,
    training_args,
    eval_size,
    sft_reward_model_prompt,
    select_range = None,
):     


    print(select_range);
    
    meta_prompts = [sft_reward_model_prompt];

    principle_collection = None;

    if data_args.principle_collection_path:
        principle_collection = {}
        with open(data_args.principle_collection_path, "r", encoding="utf-8") as f:
            principle_collection_list = json.load(f)
            for item in principle_collection_list:
                principle_collection[item["dimension"]] = item["definition"]
                if "negative_definition" in item:
                    principle_collection[item["dimension"] + "_neg"] = item[
                        "negative_definition"
                    ]

    
    ext = data_args.dataset_path.split(".")[-1]
    
    if ext in ["json", "csv"]:
        train_preference = load_dataset(ext, data_files=data_args.dataset_path)["train"]
    
        if select_range is not None:
            # If user passed a single integer, interpret it as "start index"
            if isinstance(select_range, int):
                indices = range(select_range, len(train_preference))
            else:
                # If user already passed a list or range, use it directly
                indices = select_range
            
            train_preference = train_preference.select(indices)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
    
    
        ##### Creating the reward model training dataset #####



    
    train_dataset = RewardModellingDataSet(
        data = train_preference, #done
        meta_prompts = meta_prompts, ## done
        tokenizer=tokenizer, ## done
        # principle_collection = principle_collection,
    );

    
    train_dataset, eval_dataset = split_train_into_train_and_eval(
    train_dataset=train_dataset,
    eval_size = eval_size,
    seed=200,
    )


    data_collator = DataCollatorForRewardDataset(tokenizer=tokenizer);

    return dict(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
    )
