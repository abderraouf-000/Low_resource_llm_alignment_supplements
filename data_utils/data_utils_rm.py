from dataclasses import dataclass
import json
import logging
from typing import Callable, Optional, Dict, Sequence, List, Tuple, Union

import tqdm

import einops
import pandas as pd
import torch
import transformers
from torch.nn.utils.rnn import pad_sequence
from datasets import load_dataset
from datasets import Dataset as HFDataset
from torch.utils.data import Dataset

import data_utils.common_utils as utils



BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}



def format_full_prompt(
    example: Dict[str, str], ## done
    meta_prompts: List[str], ## done
    tokenizer: transformers.PreTrainedTokenizer, ## done
    eos_token: Optional[str] = None,## done
    query_len: Optional[int] = None, ## doen
    response_len: Optional[int] = None, ## done
    output_key: str = "output", ## done
) -> str:

    if eos_token is None:
        eos_token = "";

    if "example_id" in example:
        total_meta_prompt = len(meta_prompts)
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt]
    else:
        meta_prompt = meta_prompts[0]


    if example.get("input", "") != "":
        prompt_format = BASE_PROMPT_DICT["prompt_input"]
    else:
        prompt_format = BASE_PROMPT_DICT["prompt_no_input"]

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


    return formatted_prompt





def _tokenize_fn(strings: Sequence[str], tokenizer: transformers.PreTrainedTokenizer) -> Dict:

    """Tokenize a list of strings."""


    def _tokenize(x):
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
    def __init__(self, data ,  meta_prompts,tokenizer: transformers.PreTrainedTokenizer):

        super(RewardModellingDataSet, self).__init__()

        list_dict_data = data;


        ######## Preprocess_for_reward_model #######

        ### create index tensors ###


        index_0, index_1 = tuple(
          torch.full(
        size=(len(list_dict_data), 1), fill_value=fill_value, dtype=torch.long
                )
               for fill_value in (0, 1)
        )

        ### create choice tensor ###

        choice = torch.tensor(
            [[{1: 0, 2: 1}[dict_data["preference"]]] for dict_data in list_dict_data]
        )

        ### Construct prompts for responses ###

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

        print("---- Tokenized sentence : ----\n\n",tokenized_0[0]);
        
        print("---- decoded sentence : ----\n\n", tokenizer.batch_decode(
                sequences[0].input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,))



        input_ids = [
        list(pair)
        for pair in zip(tokenized_0["input_ids"], tokenized_1["input_ids"])
        ]




        labels = [
        list(pair) for pair in zip(tokenized_0["labels"], tokenized_1["labels"])
        ]

        packaged_data = dict(
            input_ids=input_ids,
            labels=labels,
            index_0=index_0,
            index_1=index_1,
            choice=choice,
        );


        self.input_ids = packaged_data["input_ids"];
        self.labels = packaged_data["labels"];
        self.index_0 = packaged_data["index_0"];
        self.index_1 = packaged_data["index_1"];
        self.choice = packaged_data["choice"];


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
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


    def __init__(self,tokenizer : transformers.PreTrainedTokenizer):

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
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,
    skipped_steps,
):

    meta_prompts = utils.make_meta_prompts(data_args.meta_prompt_pattern);


    if data_args.dataset_path.endswith("json"):
        train_preference = load_dataset("json", data_files=data_args.dataset_path)[
            "train"
        ].skip(skipped_steps)
        
    if data_args.dataset_path.endswith("csv"):
        train_preference = load_dataset("csv", data_files=data_args.dataset_path)[
            "train"
        ].skip(skipped_steps)


    ##### Creating the reward model training dataset #####

    train_dataset = RewardModellingDataSet(
        data = train_preference, #done
        meta_prompts = meta_prompts, ## done
        tokenizer=tokenizer, ## done
        # principle_collection = principle_collection,
    );

    train_dataset, eval_dataset = split_train_into_train_and_eval(
    train_dataset=train_dataset,
    eval_size = 500,
    seed=200,
    )

    data_collator = DataCollatorForRewardDataset(tokenizer=tokenizer);

    return dict(
        train_dataset = train_dataset,
        eval_dataset = eval_dataset,
        data_collator = data_collator,
    )







