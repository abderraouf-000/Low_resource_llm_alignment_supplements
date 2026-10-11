# Copyright 2023 The Self-Align Team
# Copyright 2023 The Alpaca Team
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import copy
import dataclasses
import json
from dataclasses import dataclass
import logging
import math
import os
from pathlib import Path
import random
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import einops
import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

import accelerate
from accelerate.optimizer import AcceleratedOptimizer
from accelerate.utils import convert_outputs_to_fp32

import transformers
from transformers.trainer_utils import enable_full_determinism, set_seed

from data_utils.data_utils_ppo import QueryResponseDataset
import data_utils.common_utils as utils
from models.trainer_utils import create_optimizer, create_scheduler



if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler


FIRST_STEP_IDX = 1



class KLController(abc.ABC):
    value: Union[int, float]

    def step(self, *args, **kwargs):
        pass


class FixedKLController(KLController):
    def __init__(self, kl_coef):
        super(FixedKLController, self).__init__()
        self.value = kl_coef



class RLTrainer(object):
    def __init__(
            self,
            args,
            train_dataset : QueryResponseDataset,
            eval_dataset: QueryResponseDataset,
            data_collator : Callable,
            tokenizer: transformers.PreTrainedTokenizer,
            policy = nn.Module,
            ref_policy = nn.Module,
            reward_model = nn.Module,
            second_reward_model = nn.Module,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr_scheduler: Optional[LRScheduler] = None,
    ):
        super(RLTrainer,self).__init__();
        self.train_dataset = train_dataset;
        self.eval_dataset = eval_dataset;
        self.data_collator = data_collator;
        self.tokenizer = tokenizer;
        self.policy = policy;
        self.ref_policy = ref_policy;
        self.reward_model = reward_model;
        self.second_reward_model = second_reward_model;
        
        self.optimizer = optimizer;
        self.lr_scheduler = lr_scheduler;
        self.kl_ctl = FixedKLController(kl_coef=args.kl_coef)
        self.args = args;

        set_seed(self.args.seed); # Set seed for reproducibility

        self.policy_meta_prompts = None;
        self.reward_meta_prompts = None;
        self.principle_collection = None;
        

        @abc.abstractmethod
        @torch.inference_mode()
        def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
            raise NotImplementedError

        @abc.abstractmethod
        def compute_loss(
            self, rollouts: Dict[str, torch.Tensor]
        ) -> Tuple[torch.Tensor, Dict]:
            raise NotImplementedError 

        @abc.abstractmethod
        @torch.inference_mode()
        def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
            raise NotImplementedError


        



    ## Training On rollouts
    def step_with_rollouts(self, rollouts):

        """Based on fixed rollouts, run PPO for multiple epochs."""


        
        rollouts_dataloader = self.get_rollouts_dataloader(rollouts=rollouts);
        
        stats_list = [];
        
        for epoch_idx in range(self.args.noptepochs):

            for batch_idx, rollouts_batch in tqdm.tqdm(
                enumerate(rollouts_dataloader, 1),
                total=len(rollouts_dataloader),
                desc="training steps",
            ):
                
                stats_for_this_step = {}
                
                policy_loss, policy_stats = self.compute_policy_loss(
                    rollouts_batch
                )
                
                stats_for_this_step.update(policy_stats)
                
                policy_loss.backward();

                value_loss, value_stats = self.compute_value_loss(rollouts_batch)
               
                stats_for_this_step.update(value_stats)
                
                value_loss.backward();

                # if self.accelerator.sync_gradients:
                    # Gradient norm almost blows up at some point, but stabilizes eventually, even w/o clipping.
                if self.args.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        self.policy.parameters(), self.args.max_grad_norm
                    );
                
                # stats_for_this_step[
                #     "loss/grad_norm"
                # ] = self._compute_grad_norm()

                stats_list.append(stats_for_this_step);

                self.optimizer.step()
                
                self.optimizer.zero_grad(set_to_none=True)

        return utils.merge_dict(
            stats_list, torch.stack
        )  # list of dict -> dict: str -> 1-D tensor
    
    




    def step(self, train_dataloader, step_idx: int):

        queries_batches = [
            next(train_dataloader) for _ in range(self.args.rollout_accumulation_steps)
        ]
        
        rollouts = self.rollout(queries_batches) ## Returns the rollouts for these data samples.
        train_stats = self.step_with_rollouts(rollouts) ##  

        
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        

        stats = self.record_step_stats(
            rollouts=rollouts,
            train_stats=train_stats,
            step_idx=step_idx,
            kl_coef=self.kl_ctl.value,
        )
        
        self.kl_ctl.step(stats["objective/kl_sum_seq"])
        return stats

    
    def create_optimizer_and_scheduler(self, num_training_steps: int):
        self.optimizer = create_optimizer(
            args=self.args, model=self.policy, optimizer=self.optimizer
        )
        self.lr_scheduler = create_scheduler(
            args=self.args,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            num_training_steps=num_training_steps,
        )

        return self.optimizer, self.lr_scheduler

        
        



    
    # @torch.inference_mode()
    # def evaluate(self, step_idx: int, unwrapped_policy=None):
    #     """Evaluate by generating sequences with test prefixes.

    #     FSDP compat: all devices should do the forward pass, since sharded params need to be summoned.
    #                  only write results in the main process.
    #     """
    #     # TODO: unhardcode inference args.

        
        
    #     print(
    #         f"Start evaluation at step: {step_idx}"
    #     );

        
    #     pure_queries, prompts, list_dict_data = (
    #         self.eval_dataset.pure_queries,
    #         self.eval_dataset.prompts,
    #         self.eval_dataset.list_dict_data,
    #     )
        
        
    #     if any(item is None for item in (pure_queries, prompts, list_dict_data)):
    #         print(
    #             "No evaluation data, skipping evaluation." )
    #         return

    #     # Constants.
        
    #     model_name = Path(
    #         self.args.output_dir
    #     ).stem  # Don't use the helper in common, as no checkpoint is saved yet.
        
    #     model_name_at_step = f"{model_name}_ckpt_{step_idx}"
        
    #     temperature = 0.7
        
    #     del model_name

    #     # Start evaluation.
    #     self.policy.eval()

    #     outputs = decode_prompts_with_huggingface_given_model(
    #         model= self.policy,
    #         tokenizer=self.tokenizer,
    #         prompts=prompts,
    #         decoding_args = HFDecodingArguments(
    #             max_new_tokens=self.args.response_len, temperature=temperature, 
    #         ),
    #         per_device_batch_size = self.args.per_device_eval_batch_size,
    #     )


        
    #     sequences = [i + o for i, o in utils.zip_(prompts, outputs)]


    #     # print(f"\n\n\n----------------------------------------Reward model sequences are : {sequences} \n\n\n");

    #     # print(f" -- --  - -- - - - - - - - - - Pad token ID: {self.tokenizer.pad_token_id} - -- - - - -  - -- \n")
        
    #     # print(f"Pure queries are : {pure_queries, outputs}")

    #     unpadded_queries = []
    #     for query in pure_queries:

    #         mask = query != self.tokenizer.pad_token_id
    #         unpadded_query = query[mask]
    #         unpadded_queries.append(unpadded_query)
        
    #     text_pure_queries = self.tokenizer.batch_decode(
    #         unpadded_queries,
    #         skip_special_tokens=True,
    #         clean_up_tokenization_spaces=False,
    #     )

        
    #     sequences = [
    #         self.prepare_reward_inputs(inputs=q, outputs=r,eos_token = self.tokenizer.eos_token)
    #         for q, r in utils.zip_(text_pure_queries, outputs)
            
    #     ]


        

        
    #     rewards = score_sequences_with_huggingface_given_model(
    #         model=self.reward_model,
    #         tokenizer=self.tokenizer,
    #         sequences=sequences,
    #         per_device_batch_size=self.args.rollout_per_device_batch_size,
    #         divide_work=False,
    #     )

        

    #     results = [
    #         {"reward": reward, model_name_at_step: output, **example}
    #         for reward, output, example in utils.zip_(
    #             rewards, outputs, list_dict_data
    #         )
    #     ]

        
    #     if self.args.output_dir is not None:
            
    #         with open(
    #             os.path.join(self.args.output_dir, f"eval_results_{step_idx}.json"),
    #             "w",
    #             encoding="utf-8",
    #         ) as f:
    #             json.dump(results, f, indent=2)

    #     print(
    #         f"End evaluation at step: {step_idx}. Processed {len(results)} examples"
    #     );


    


    
    @torch.inference_mode()
    def evaluate(self, step_idx: int, unwrapped_policy=None, best_of_n: int = 5):
        """Evaluate by generating sequences with test prefixes using Best-of-N sampling.
        For each prompt, generate N candidate responses and select the one with the highest reward.
        FSDP compat: all devices should do the forward pass, since sharded params need to be summoned.
                     only write results in the main process.
        """
        print(f"Start evaluation at step: {step_idx} with Best-of-N={best_of_n}")
    
        pure_queries, prompts, list_dict_data = (
            self.eval_dataset.pure_queries,
            self.eval_dataset.prompts,
            self.eval_dataset.list_dict_data,
        )
    
        if any(item is None for item in (pure_queries, prompts, list_dict_data)):
            print("No evaluation data, skipping evaluation.")
            return
    
        model_name = Path(self.args.output_dir).stem
        model_name_at_step = f"{model_name}_ckpt_{step_idx}"
        temperature = 0.7
        del model_name
    
        # Decode unpadded text queries once (used for reward model input construction)
        unpadded_queries = []
        for query in pure_queries:
            mask = query != self.tokenizer.pad_token_id
            unpadded_queries.append(query[mask])
    
        text_pure_queries = self.tokenizer.batch_decode(
            unpadded_queries,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    
        # --- Best-of-N: generate N candidates per prompt ---
        self.policy.eval()
    
        # Shape: all_candidate_outputs[n] = list of outputs (one per prompt) for the n-th sample
        all_candidate_outputs = []
        for n in range(best_of_n):
            candidate_outputs = decode_prompts_with_huggingface_given_model(
                model=self.policy,
                tokenizer=self.tokenizer,
                prompts=prompts,
                decoding_args=HFDecodingArguments(
                    max_new_tokens=self.args.response_len, temperature=temperature
                ),
                per_device_batch_size=self.args.per_device_eval_batch_size,
            )
            all_candidate_outputs.append(candidate_outputs)
    
        # Score all candidates: flatten -> score -> unflatten
        # Interleave by prompt: [prompt0_n0, prompt0_n1, ..., prompt1_n0, ...]
        # This makes it easy to group by prompt index later.
        flat_sequences = [
            self.prepare_reward_inputs(inputs=q, outputs=candidate_outputs[i], eos_token = self.tokenizer.eos_token)
            for i, q in enumerate(text_pure_queries)
            for candidate_outputs in all_candidate_outputs
        ]
    
        flat_rewards = score_sequences_with_huggingface_given_model(
            model=self.reward_model,

            
            tokenizer=self.tokenizer,
            sequences=flat_sequences,
            per_device_batch_size=self.args.rollout_per_device_batch_size,
            divide_work=False,
        )
    
        # --- Select the best candidate per prompt ---
        best_outputs = []
        best_rewards = []
        all_candidate_rewards = []  # shape: [num_prompts, best_of_n] — logged for analysis
    
        for prompt_idx in range(len(prompts)):
            # Rewards for all N candidates of this prompt
            candidate_rewards = flat_rewards[prompt_idx * best_of_n : (prompt_idx + 1) * best_of_n]
            candidate_outputs_for_prompt = [all_candidate_outputs[n][prompt_idx] for n in range(best_of_n)]
    
            best_n_idx = int(torch.tensor(candidate_rewards).argmax())
            best_outputs.append(candidate_outputs_for_prompt[best_n_idx])
            best_rewards.append(candidate_rewards[best_n_idx])
            all_candidate_rewards.append(candidate_rewards)
    
        # --- Build results ---
        results = [
            {
                "reward": best_reward,
                "all_candidate_rewards": candidate_rewards,  # useful for analysis
                model_name_at_step: best_output,
                **example,
            }
            for best_reward, best_output, candidate_rewards, example in utils.zip_(
                best_rewards, best_outputs, all_candidate_rewards, list_dict_data
            )
        ]
    
        if self.args.output_dir is not None:
            out_path = os.path.join(self.args.output_dir, f"eval_results_{step_idx}.json")
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
    
        print(f"End evaluation at step: {step_idx}. Processed {len(results)} examples")











    

    

    def train(self, resume_training_ckpt = None):
        
        total_epochs = self.args.total_epochs;
        total_episodes = len(self.train_dataset) * total_epochs;
        total_steps = total_episodes // self.args.rollout_batch_size;
        print(f"***Training starts***\n" + 
        f"Total epochs: {total_epochs} => Total episodes: {total_episodes} => Total steps: {total_steps}");
        
        self.create_optimizer_and_scheduler(total_steps)

        ## skipping last training steps
        skipping_steps = 0
        if resume_training_ckpt is not None:
            skipping_steps = self.resume_training(resume_training_ckpt)
            print(
                f"Resuming training from {resume_training_ckpt} at step {skipping_steps}."
            )
        train_dataloader = self.get_train_dataloader();

        for step_idx in tqdm.tqdm(
            range(FIRST_STEP_IDX,total_steps + FIRST_STEP_IDX),
            desc = "Training steps",
            total = total_steps,
        ):

            
            if step_idx % self.args.save_steps == 0:
                # if step_idx > skipping_steps:
                self.save_model(
                    os.path.join(self.args.output_dir, f"ppo-checkpoint-{step_idx}")
                )

                    
            if not self.args.eval_steps == None and step_idx % self.args.eval_steps == 0 :
                self.evaluate(step_idx);



                
            self.step(train_dataloader,step_idx);

        




    



    def get_rollouts_dataloader(
    self, rollouts: Dict[str, torch.Tensor], shuffle=False, drop_last=True, keys=None
                        ):
        if keys is None:
            keys = tuple(rollouts.keys())



        def collate_rollouts(instances: Sequence[tuple]):
            return {
                key: torch.stack([instance[idx] for instance in instances])
                for idx, key in enumerate(keys)
            }

        rollouts_dataset = TensorDataset(*[rollouts[key] for key in keys])

        rollouts_dataloader = DataLoader(
            dataset=rollouts_dataset,
            batch_size=self.args.step_per_device_batch_size,
            collate_fn=collate_rollouts,
            shuffle=shuffle,
            drop_last=drop_last,
        )
        return rollouts_dataloader
    



    

    
    def _log_batch_size(self, loader: DataLoader, loader_name):
        batch = next(iter(loader))
        if isinstance(batch, torch.Tensor):
            batch_size = batch.shape[0]
        elif isinstance(batch, (list, tuple)):
            batch_size = batch[0]
        else:
            tensor = list(batch.values())[0]
            batch_size = tensor.size(0)
        print(
            f"Batch size of {loader_name} dataloader: {batch_size}"
        )



    

    def get_train_dataloader(self):

            train_dataloader = DataLoader(
                dataset=self.train_dataset,
                collate_fn=self.data_collator,
                batch_size= self.args.rollout_per_device_batch_size, ## set to 1 for the moment.
                shuffle=False,
                drop_last=True,
            )
            self._log_batch_size(train_dataloader, "train_dataloader")
            return utils.InfiniteLoader(train_dataloader);


        
        

    def get_policy_meta_prompts(self):
        if self.policy_meta_prompts is not None:
            return self.policy_meta_prompts

        policy_meta_prompt_pattern = self.args.data_config.policy_meta_prompt_pattern
        policy_meta_prompts = utils.make_meta_prompts(
            policy_meta_prompt_pattern,
        )
        
        self.policy_meta_prompts = policy_meta_prompts
        return policy_meta_prompts;




        
        

    def get_reward_meta_prompts(self):
        
        if self.reward_meta_prompts is not None:
            return self.reward_meta_prompts

        reward_meta_prompt_pattern = self.args.data_config.reward_meta_prompt_pattern
        reward_meta_prompts = utils.make_meta_prompts(
            reward_meta_prompt_pattern,
        )
        self.reward_meta_prompts = reward_meta_prompts
        return reward_meta_prompts;
        



    
    def get_principle_collection(self):
        if self.principle_collection is not None:
            return self.principle_collection

        principle_collection_path = self.args.data_config.principle_collection_path
        assert os.path.exists(principle_collection_path)
        print("Loading principle collection from", principle_collection_path)
        with open(principle_collection_path, "r", encoding="utf-8") as f:
            principle_collection = json.load(f)
        self.principle_collection = principle_collection
        return principle_collection

        
    
    
    def prepare_reward_inputs(self, inputs, outputs,eos_token ,question_type_flag="GENERAL"):
    
        # print(f"prepare_reward_inputs : \n\n\n {inputs} \n\n\n\n {outputs} \n\n\n\n ")
    
       
        reward_meta_prompt = '\n' + self.get_reward_meta_prompts()[0] + '\n';
    
        helpfulness_keywords = ["honest and accurate", "educational and engaging", "comprehensive"];
    
        redteaming_keywords = ["ethical", "privacy Protection"];


        
        if "{Dimensions}" in reward_meta_prompt:
    
            principle_collection = self.get_principle_collection()
            
            # Filter to only include principles with helpfulness or redteaming keywords
            
            filtered_principles = []
            for item in principle_collection:
                helpfulness_principle_flag = False
                redteaming_principle_flag = False
                
                for keyword in helpfulness_keywords:
                    if keyword in item["dimension"].lower():
                        helpfulness_principle_flag = True
                        
                for keyword in redteaming_keywords:
                    if keyword in item["dimension"].lower():
                        redteaming_principle_flag = True
                        
                # Only include principles that match our target keywords
                if helpfulness_principle_flag or redteaming_principle_flag:
                    filtered_principles.append(item)
            dimension_str = []
            for item in filtered_principles:
                dimension_str.append(f"- {item['definition']}");                
    
            # Apply max_principles limit if specified
            if self.args.data_config.max_principles is not None:
                dimension_str = dimension_str[:self.args.data_config.max_principles]
    
            dimension_str = "\n".join(dimension_str)
            
            reward_meta_prompt = reward_meta_prompt.replace(
                "{Dimensions}", dimension_str
            )
        
        
        return (
            reward_meta_prompt.format(
            Input=inputs,
            Output=outputs,
        )
        + eos_token
               )


        

    # def prepare_reward_inputs(self, inputs, outputs, question_type_flag="GENERAL"):

    #     # print(f"prepare_reward_inputs : \n\n\n {inputs} \n\n\n\n {outputs} \n\n\n\n ")

        
    #     reward_meta_prompt = self.get_reward_meta_prompts()[0]

    #     helpfulness_keywords = ["honest and accurate", "educational and engaging", "comprehensive"];
    #     helpfulness_bonus_weight = 128.0;

    #     redteaming_keywords = ["ethical", "privacy Protection"];
    #     redteaming_bonus_weight = 128.0;

    #     if "{Dimensions}" in reward_meta_prompt:
    #         principle_collection = self.get_principle_collection()
    #         random.shuffle(principle_collection)
    #         dimension_str = []
    #         for item in principle_collection:
    #             dimension_str.append(f"- {item['definition']}")
    #         if self.args.data_config.max_principles is not None:
    #             if "weight" not in principle_collection[0]:
    #                 dimension_str = dimension_str[
    #                     : self.args.data_config.max_principles
    #                 ]
    #             else:
    #                 remaining_weights = []

    #                 for item in principle_collection:
                        
    #                     assert "weight" in item
                        
    #                     helpfulness_principle_flag = False;
                        
    #                     for keyword in helpfulness_keywords:
                            
    #                         if keyword in item["dimension"].lower():
    #                             helpfulness_principle_flag = True

    #                     redteaming_principle_flag = False;
                        
    #                     for keyword in redteaming_keywords:
    #                         if keyword in item["dimension"].lower():
    #                             redteaming_principle_flag = True

    #                     if (
    #                         question_type_flag == "DONOTANSWER"
    #                         and helpfulness_principle_flag
    #                     ):
    #                         remaining_weights.append(
    #                             item["weight"] + helpfulness_bonus_weight
    #                         )

                            
    #                     elif(
    #                         question_type_flag == "REDTEAMING" or question_type_flag == "DONOTANSWER" 
    #                         and redteaming_principle_flag
    #                     ):
                            
    #                         remaining_weights.append(
    #                             item["weight"] + redteaming_bonus_weight
    #                         )
                            
    #                     else:
    #                         remaining_weights.append(item["weight"]);

    #                 remaining_idx = list(range(len(dimension_str)))

    #                 sampled_dimension_str = []
    #                 while (
    #                     len(sampled_dimension_str)
    #                     < self.args.data_config.max_principles
    #                 ):
    #                     sampled_idx = random.choices(
    #                         list(range(len(remaining_idx))), weights=remaining_weights
    #                     )[0]
    #                     sampled_dimension_str.append(
    #                         dimension_str[remaining_idx[sampled_idx]]
    #                     )
    #                     remaining_idx.pop(sampled_idx)
    #                     remaining_weights.pop(sampled_idx)

    #                 dimension_str = sampled_dimension_str

    #         dimension_str = "\n".join(dimension_str);
            
    #         reward_meta_prompt = reward_meta_prompt.replace(
    #             "{Dimensions}", dimension_str
    #         )

    #     print(f"reward_meta_prompt after formatting { reward_meta_prompt.format(Input=inputs,Output=outputs)}");
        
        
    #     return reward_meta_prompt.format(
    #         Input=inputs,
    #         Output=outputs,
    #     );



    

    @abc.abstractmethod
    @torch.inference_mode()
    def save_model(self, output_dir: Optional[str] = None):
        raise NotImplementedError

    @abc.abstractmethod
    @torch.inference_mode()
    def resume_training(self, checkpoint_dir: str):
        raise NotImplementedError
    

    def truncate_after_eos(completions, eos_token_id):
        # We truncate tokens after eos_token_id
        clean_completions = completions.tolist()
        for idx, completion in enumerate(clean_completions):
            try:
                end_idx = completion.index(eos_token_id)
                clean_completions[idx] = completion[:end_idx]
            except ValueError:
                pass
        return clean_completions


    

def cast_with_native_amp(
    func: Callable, mixed_precision: Optional[str] = None
) -> Callable:
    """Almost like how huggingface accelerate cast `model.forward`."""
    if mixed_precision not in ("fp16", "bf16"):
        logger.warning(
            f"Unknown mixed precision mode: {mixed_precision}, falling back to fp32."
        )
        return func

    if mixed_precision == "fp16":
        output_func = torch.cuda.amp.autocast(dtype=torch.float16)(func)
    else:
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        output_func = torch.autocast(device_type=device_type, dtype=torch.bfloat16)(
            func
        )
    output_func = convert_outputs_to_fp32(output_func)
    return output_func



def truncate_after_eos(completions, eos_token_id):
    # We truncate tokens after eos_token_id
    clean_completions = completions.tolist()
    for idx, completion in enumerate(clean_completions):
        try:
            end_idx = completion.index(eos_token_id)
            clean_completions[idx] = completion[:end_idx]
        except ValueError:
            pass
    return clean_completions
        




        










import torch
import tqdm
import gc
from typing import List, Union, Optional, Dict
from dataclasses import dataclass
from transformers import PreTrainedModel, PreTrainedTokenizer, LlamaTokenizerFast
import math
import data_utils.common_utils as common_utils


@dataclass
class HFDecodingArguments:
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 1.0
    top_k: int = 0
    do_sample: bool = True
    num_return_sequences: int = 1
    pad_token_id: Optional[int] = None
    eos_token_id: Optional[int] = None





@torch.inference_mode()
def decode_prompts_with_huggingface_given_model(
    model,  # Policy model (has respond method)
    tokenizer: PreTrainedTokenizer,
    prompts: List[str],
    decoding_args: HFDecodingArguments,
    per_device_batch_size: int = 1,
) -> List[str]:
    """
    Decode prompts using a Policy model, following the same logic as rollout method.
    """
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    all_outputs = []
    
    # Convert prompts to the same format as your rollout queries_data
    queries_data = []
    for i in range(0, len(prompts), per_device_batch_size):
        batch_prompts = prompts[i:i + per_device_batch_size]
        
        # Tokenize prompts to create queries (following your rollout pattern)
        tokenized = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=True
        )
        
        batch_dict = {
            "queries": tokenized.input_ids,
            "query_attn_masks": tokenized.attention_mask,
            "pure_queries": tokenized.input_ids,  # Same as queries for evaluation
            "length_bonus": torch.ones(len(batch_prompts)),  # Default length bonus
            "question_types": torch.zeros(len(batch_prompts)),  # Default to GENERAL type
        }
        queries_data.append(batch_dict)
    
    # Process each batch following your rollout logic
    for batch_idx, batch in tqdm.tqdm(
        enumerate(queries_data),
        total=len(queries_data),
        desc="decode_prompts",
    ):
        gc.collect()
        torch.cuda.empty_cache()
        
        # Unpack batch exactly like in rollout
        (
            pure_queries,
            queries,
            query_attn_masks,
            length_bonus_multiplier,
            question_types,
        ) = common_utils.unpack_dict(
            common_utils.prepare_inputs(batch, device=device),
            keys=(
                "pure_queries",
                "queries", 
                "query_attn_masks",
                "length_bonus",
                "question_types",
            ),
        )
        
        # print(f"\n\n decode queries : {tokenizer.batch_decode(queries, skip_special_tokens=True, clean_up_tokenization_spaces=False)} \n\n")
        
        # Generate responses using the same method as rollout
        respond_outputs = model.respond(
            queries, query_attn_masks, temperature=decoding_args.temperature
        );
        
        (responses,) = common_utils.unpack_dict(respond_outputs, ("responses",));

        
        # Decode responses following your rollout logic
        text_responses = tokenizer.batch_decode(
            responses,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        );
        
        
        # print(f"The response is : {text_responses} \n")
        
        # Handle special tokens exactly like in rollout
        if isinstance(tokenizer, LlamaTokenizerFast):
            llama_fast_special_tokens = [tokenizer.eos_token]
            if tokenizer.eos_token != tokenizer.bos_token:
                llama_fast_special_tokens.append(tokenizer.bos_token)
            for special_token in llama_fast_special_tokens:
                if special_token is not None:
                    text_responses = [
                        text_response.replace(special_token, f" {special_token}")
                        for text_response in text_responses
                    ]
        
        # Handle stop tokens exactly like in rollout
        parsed_stop_token = tokenizer.eos_token  # Default to eos_token for evaluation
        
        text_responses = [
            truncate_after_stop_token(text_response, parsed_stop_token).split(
                tokenizer.pad_token
            )[0]
            for text_response in text_responses
        ]
        
        # Clean responses
        cleaned_responses = [
            clean_after_stop_token(response, parsed_stop_token)
            for response in text_responses
        ]
        
        all_outputs.extend(cleaned_responses)
        
        # Clean up variables like in rollout
        del pure_queries, queries, responses
    
    return all_outputs




@torch.inference_mode() 
def score_sequences_with_huggingface_given_model(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    sequences: List[str],
    per_device_batch_size: int = 1,
    divide_work: bool = False,
) -> List[float]:

    
    """
    Score sequences using reward model, following the same logic as rollout method.
    """


    
    model.eval();
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
    
    all_rewards = []
    
    # Process in batches following rollout pattern
    for batch_start in tqdm.tqdm(
        range(0, len(sequences), per_device_batch_size),
        desc="score_sequences",
        total=math.ceil(len(sequences) / per_device_batch_size)
    ):
        
        gc.collect()
        torch.cuda.empty_cache()
        
        batch_end = min(batch_start + per_device_batch_size, len(sequences))
        batch_sequences = sequences[batch_start:batch_end]
        
        # Tokenize sequences exactly like in rollout
        sequences_tokenized = tokenizer(
            batch_sequences, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        )
        
        sequences_prepared = common_utils.prepare_inputs(
            sequences_tokenized, device=device
        )
        
        batch_size_per_device = len(batch_sequences)
        # print(f"score batch size : \n {batch_size_per_device}")
        
        # Handle sub-batching like in rollout (though usually not needed for scoring)
        sub_batch_size = per_device_batch_size  # Can be made configurable
        
        if sub_batch_size is None or sub_batch_size == batch_size_per_device:

            # print(f"sequences_prepared.input_ids{sequences_prepared.input_ids} \n\n\n {sequences_prepared.attention_mask} ")
            reward_outputs = model(
                input_ids = sequences_prepared.input_ids,
                attention_mask = sequences_prepared.attention_mask,
            )
            # print(f"reward_outputs are : {reward_outputs}")
        else:
            assert batch_size_per_device % sub_batch_size == 0
            
            reward_outputs_list = []
            
            for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                idx_start = sub_batch_idx * sub_batch_size
                idx_end = (sub_batch_idx + 1) * sub_batch_size
                sub_batch_reward_outputs = model(
                    input_ids=sequences_prepared.input_ids[idx_start:idx_end],
                    attention_mask=sequences_prepared.attention_mask[idx_start:idx_end],
                )
                reward_outputs_list.append(sub_batch_reward_outputs)
            
            reward_outputs = common_utils.merge_dict(
                reward_outputs_list, merge_fn=torch.cat
            )
            del reward_outputs_list
            del sub_batch_reward_outputs
            
            print(f"Reward outputs{reward_outputs}")
            print("=" * 20)
        
        # Extract rewards following your reward model output format
        if hasattr(reward_outputs, 'rewards'):
            batch_rewards = reward_outputs.rewards
        elif isinstance(reward_outputs, dict) and 'rewards' in reward_outputs:
            batch_rewards = reward_outputs['rewards']
        elif hasattr(reward_outputs, 'logits'):
            batch_rewards = reward_outputs.logits.squeeze(-1)
            if batch_rewards.dim() > 1:
                batch_rewards = batch_rewards[:, -1]  # Take last token
        else:
            # Fallback: assume the output tensor is the reward directly
            if isinstance(reward_outputs, torch.Tensor):
                batch_rewards = reward_outputs.squeeze(-1)
                if batch_rewards.dim() > 1:
                    batch_rewards = batch_rewards[:, -1]
            else:
                raise ValueError(f"Unsupported reward model output format: {type(reward_outputs)}")
        
        # Convert to CPU and list like in rollout
        if isinstance(batch_rewards, torch.Tensor):
            batch_rewards = batch_rewards.cpu().tolist()
        
        if isinstance(batch_rewards, (int, float)):
            batch_rewards = [batch_rewards]
        
        all_rewards.extend(batch_rewards)
    
    return all_rewards


# Helper functions exactly matching your rollout code
def truncate_after_stop_token(text: str, stop_token: str) -> str:
    """Truncate text after the first occurrence of stop_token."""
    if stop_token in text:
        return text.split(stop_token)[0] + stop_token
    return text


def clean_after_stop_token(text: str, stop_token: str = None) -> str:
    """Clean text after stop token, removing the stop token itself."""
    if stop_token is None:
        # return text.strip()
        return text
    
    if stop_token in text:
        return text.split(stop_token)[0].strip()
    return text.strip()
