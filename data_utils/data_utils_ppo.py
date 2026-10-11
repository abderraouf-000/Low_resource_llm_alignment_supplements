

import dataclasses
from typing import Callable, Dict, Optional, List, Sequence

import logging
import pandas as pd

import torch
from torch.utils.data import Dataset

import transformers
import datasets
import random

import data_utils.common_utils as utils


BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}








def format_input_and_prompt(
    example: Dict[str, str],
    meta_prompts: List[str],
) -> str:
    assert (
        "instruction" in example and "input" in example
    ), "Internal error: example missing required keys."

    
    if "example_id" in example:
        total_meta_prompt = len(meta_prompts);
        meta_prompt = meta_prompts[int(example["example_id"]) % total_meta_prompt];

    else:
        meta_prompt = meta_prompts[0];

    
    # if example.get("input", "") != "":
    #     prompt_format = BASE_PROMPT_DICT["prompt_input"];
    # else:
    #     prompt_format = BASE_PROMPT_DICT["prompt_no_input"];

    prompt_format = (
        BASE_PROMPT_DICT["prompt_input"]
        if (example.get("input") or "").strip()
        else BASE_PROMPT_DICT["prompt_no_input"]
    )
    
    
    formatted_input = prompt_format.format(**example);

    
    
    meta_prompt = meta_prompt.split("{Output}")[0]
    

    
    formatted_prompt = meta_prompt.format(Input=formatted_input);

        

    return formatted_input, formatted_prompt;








structural_class_to_idx = {
"List": 0,
"Dashed-list": 1,
"Numbered-list": 2,
"Comma-list":  3,
"Table": 4,
"Questions": 5,
"Mathematics": 6,
"Code": 7,
"Number": 8,
"Dialog": 9,
"Prose": 10,
}














class QueryResponseDataset(Dataset):
    """Dataset that emits tokenized left-padded queries."""

    def __init__(
        self,
        df: pd.DataFrame,
        meta_prompts: List[str],
        tokenizer: transformers.PreTrainedTokenizer,
        query_len: int,
        df_postprocessor: Optional[Callable] = None,
    ):

        
        super(QueryResponseDataset, self).__init__()

        
        if df_postprocessor is not None:
            df = df_postprocessor(df)


            
        list_dict_data = df.to_dict(orient="records")

        
        # prompts are strings; queries are tensors.
        
        inputs_and_prompts = [
            format_input_and_prompt(example=dict_data, meta_prompts=meta_prompts)
            for dict_data in list_dict_data
        ]

        
        formated_inputs, prompts = zip(*inputs_and_prompts);    
        

        length_bonus = [
            dict_data.get("length_bonus", 1.0) for dict_data in list_dict_data
        ]

        # For question_type:
        # 0: general
        # 1: reasoning
        # 2: red-teaming


        
        question_types = [
            dict_data.get("question_type", 0) for dict_data in list_dict_data
        ]

        
        structural_classes = [
            structural_class_to_idx[dict_data.get("structural_class","Prose")] for dict_data in list_dict_data
        ]

        ground_truth_outputs = [
            tokenizer(dict_data.get("output",""), return_tensors="pt", truncation=False,add_special_tokens=False).input_ids.squeeze(
                dim=0
            )
            for dict_data in list_dict_data 
        ]
        

        pure_queries = [
            tokenizer(
                formated_input, return_tensors="pt", truncation=False, add_special_tokens=False
            ).input_ids.squeeze(dim=0)
            for formated_input in formated_inputs
        ];
        

        queries = [
            tokenizer(prompt, return_tensors="pt", truncation=False,add_special_tokens=False).input_ids.squeeze(
                dim=0
            )
            for prompt in prompts
        ]
        

        filtered_inputs = []
        filtered_queries = []
        filtered_length_bonus = []
        filtered_question_types = []
        filtered_structural_classes = []
        filtered_ground_truth_outputs = []

        

        for pure_query, query, ex_length_bonus, ex_question_type, ex_structural_class,ground_truth_output in zip(
            pure_queries, queries, length_bonus, question_types,structural_classes, ground_truth_outputs
        ):
            
            if len(query) <= query_len:
                filtered_inputs.append(pure_query)
                filtered_queries.append(query)
                filtered_length_bonus.append(ex_length_bonus)
                filtered_question_types.append(ex_question_type)
                filtered_structural_classes.append(ex_structural_class)
                filtered_ground_truth_outputs.append(ground_truth_output)


        pure_queries = torch.stack(
            [
                utils.left_pad(
                    pure_query, target_size=(query_len,), value=tokenizer.pad_token_id
                )
                for pure_query in filtered_inputs
            ]
        )

        queries = torch.stack(
            [
                utils.left_pad(
                    query, target_size=(query_len,), value=tokenizer.pad_token_id
                )
                for query in filtered_queries
            ]
        )
        
        ground_truth_outputs = torch.stack(
            [
                utils.left_pad(
                    filtered_ground_truth_output, target_size=(query_len,), value=tokenizer.pad_token_id
                )
                for filtered_ground_truth_output in filtered_ground_truth_outputs
            ]
        )

        self.length_bonus = torch.tensor(
            filtered_length_bonus, dtype=torch.float32
        ).reshape(-1, 1);
        
        self.question_types = torch.tensor(
            filtered_question_types, dtype=torch.long
        ).reshape(-1, 1);
        
        self.structural_classes = torch.tensor(
            filtered_structural_classes, dtype=torch.long
        ).reshape(-1, 1);
        
        self.ground_truth_outputs = ground_truth_outputs
        self.pure_queries = pure_queries
        self.queries = queries
        self.query_attn_masks = queries.ne(tokenizer.pad_token_id).long()

        assert self.pure_queries.shape[0] == self.queries.shape[0]
        assert self.pure_queries.shape[0] == self.query_attn_masks.shape[0]
        assert self.pure_queries.shape[0] == self.length_bonus.shape[0]
        assert self.pure_queries.shape[0] == self.question_types.shape[0]
        assert self.pure_queries.shape[0] == self.structural_classes.shape[0]
        assert self.pure_queries.shape[0] == self.ground_truth_outputs.shape[0]

        # Auxiliary data.
        self.prompts = prompts
        self.list_dict_data = list_dict_data

    def __getitem__(self, i):
        return_dict = dict(
            pure_queries=self.pure_queries[i],
            queries=self.queries[i],
            query_attn_masks=self.query_attn_masks[i],
            length_bonus=self.length_bonus[i],
            question_types=self.question_types[i],
            structural_classes = self.structural_classes[i],
            ground_truth_outputs = self.ground_truth_outputs[i],
        )

        # print("++++++++++++++ return_dict ++++++++++++++++++", return_dict);
                
        return return_dict

    def __len__(self):
        return len(self.queries)






@dataclasses.dataclass
class DataCollatorForQueryResponseDataset(object):
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        return {
            key: torch.stack([instance[key] for instance in instances])
            for key in instances[0].keys()
        }







def make_rl_data_module(


    
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    training_args,


):

    
    policy_meta_prompts = utils.make_meta_prompts(data_args.policy_meta_prompt_pattern)

    if data_args.dataset_path.endswith("json"):
        train_instructions = datasets.load_dataset(
            "json", data_files=data_args.dataset_path
        )
    else:
        train_instructions = datasets.load_dataset(
            data_args.dataset_path, data_args.dataset_name
        )


        
    train_df = pd.concat(
        [pd.DataFrame(train_instructions[split]) for split in data_args.train_splits]
    )

    train_df = train_df.iloc[training_args.skipped_data:].reset_index(drop=True)


    
    if data_args.eval_dataset_path.endswith("json"):
        
        eval_instructions = datasets.load_dataset(
            "json", data_files=data_args.eval_dataset_path
        )
        
    else:
        
        eval_instructions = datasets.load_dataset(
            data_args.eval_dataset_path, data_args.dataset_name
        )
        
        
    eval_df = pd.concat(
        [pd.DataFrame(eval_instructions[split]) for split in data_args.eval_splits]
    );

    
    eval_df = eval_df.sample(n=10, random_state=42);
    
    

    train_dataset = QueryResponseDataset(
        df=train_df,
        meta_prompts=policy_meta_prompts,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
    )

    
    
    eval_dataset = QueryResponseDataset(
        df=eval_df,
        meta_prompts=policy_meta_prompts,
        tokenizer=tokenizer,
        query_len=training_args.query_len,
    );

    
    
    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForQueryResponseDataset(),
    )






















