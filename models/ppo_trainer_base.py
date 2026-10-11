import json
import gc
import glob
from itertools import chain
import logging
import os
import pathlib
import re
from typing import Callable, Dict, List, Optional, Tuple, Union

import accelerate

import numpy as np

import pandas as pd

import torch

import tqdm

import transformers

from transformers import LlamaTokenizerFast

from peft.utils import WEIGHTS_NAME, get_peft_model_state_dict

from data_utils.data_utils_ppo import QueryResponseDataset

import data_utils.common_utils as common_utils

import models.rl_models as rl_models

from models.qlora_model import load_4bit_model_for_inference

from models.reward_model import load_4bit_reward_model_for_inference

from models.rl_trainer import RLTrainer, truncate_after_eos

from transformers import LlamaForCausalLM





if torch.__version__ < "2.0.0":
    LRScheduler = torch.optim.lr_scheduler._LRScheduler  # noqa
else:
    LRScheduler = torch.optim.lr_scheduler.LRScheduler






ADAPTER_MODEL_DIR = "adapter_model";
TRAINING_ARGS_NAME = "training_args.bin";
TRAINER_STATE_NAME = "trainer_state.json";
OPTIMIZER_NAME = "optimizer.pt";
SCHEDULER_NAME = "scheduler.pt";
VALUE_HEAD_NAME = "value_head.pt";
SCALER_NAME = "scaler.pt";






idx_to_structural_class = {
    0: "List",
    1: "Dashed-list",
    2: "Numbered-list",
    3: "Comma-list",
    4: "Table",
    5: "Questions",
    6: "Mathematics",
    7: "Code",
    8: "Number",
    9: "Dialog",
    10: "Prose",
}













from collections import defaultdict



from rule_based_contrastive_sampling.utils import *



        
import logging
import sys

class PPOTrainer(RLTrainer):
    
    def __init__(
            self,
            args,
            train_dataset : QueryResponseDataset,
            eval_dataset: QueryResponseDataset,
            data_collator: Callable,
            policy: rl_models.ActorCritic,
            ref_policy: rl_models.Policy,
            reward_model,
            second_reward_model,
            tokenizer: transformers.PreTrainedTokenizer,
            optimizer: Optional[torch.optim.Optimizer] = None,
            lr_scheduler: Optional[LRScheduler] = None,
    ):

        
        super(PPOTrainer, self).__init__(
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            policy=policy,
            ref_policy=ref_policy,
            reward_model=reward_model,
            second_reward_model=second_reward_model,
            tokenizer=tokenizer,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
        );


        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu");
        self.reward_normalizer = RewardNormalizer();
        
        self.setup_logger(log_level=logging.INFO);
        




    def setup_logger(self, log_level=logging.INFO, log_file=None):
        """
        Sets up self.logger for the class.
    
        Args:
            log_level: logging.INFO, DEBUG, etc.
            log_file: Optional path to save logs to a file.
        """
    
        self.logger = logging.getLogger("RolloutLogger")
        self.logger.setLevel(log_level)
        self.logger.propagate = False  # prevent double logs
    
        # Clear previous handlers
        if self.logger.hasHandlers():
            self.logger.handlers.clear()
    
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(log_level)
        ch_formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S"
        )
        ch.setFormatter(ch_formatter)
        self.logger.addHandler(ch)
    
        # Optional file handler
        if log_file is not None:
            fh = logging.FileHandler(log_file)
            fh.setLevel(log_level)
            fh_formatter = logging.Formatter(
                "%(asctime)s | %(levelname)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
            )
            fh.setFormatter(fh_formatter)
            self.logger.addHandler(fh)





      
    def _shape_reward(
            self,
            rewards: torch.Tensor,
            responses: torch.Tensor,
            logprobs: torch.Tensor,
            ref_logprobs: torch.Tensor,
            length_bonus: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Shape rewards with KL penalty and length bonus.
    
        Returns:
            Dict with keys:
                shaped_rewards, non_score_rewards, kl
        """
        
        # --- Compute KL divergence approximation ---
        if self.args.kl_approximator == "k1":
            # KL(q||p) ≈ max(log q - log p, 0)
            kl = torch.clamp(logprobs - ref_logprobs, min=0.0)
        elif self.args.kl_approximator == "k3":
            # KL(q||p) ≈ exp(log(p/q)) - 1 - log(p/q)
            log_r = ref_logprobs - logprobs
            kl = torch.exp(log_r) - 1.0 - log_r
        else:
            raise ValueError(f"Unknown KL approximator: {self.args.kl_approximator}")
    
        self.logger.debug(
            "_shape_reward: KL stats - mean: %.5f, max: %.5f, min: %.5f",
            kl.mean().item(), kl.max().item(), kl.min().item()
        )
    
        # --- Compute non-score rewards (KL penalty) ---
        non_score_rewards = -self.kl_ctl.value * kl
        self.logger.debug(
            "_shape_reward: Non-score rewards stats - mean: %.5f, max: %.5f, min: %.5f",
            non_score_rewards.mean().item(), non_score_rewards.max().item(), non_score_rewards.min().item()
        )
    
        # --- Add terminal rewards + length bonus ---
        shaped_rewards = non_score_rewards.clone()
        
        shaped_rewards[:, -1] += rewards + (
            torch.clamp(length_bonus, max=self.args.length_bonus_upper_bound)
            * self.args.length_bonus_score
        )
    
        self.logger.debug(
            "_shape_reward: Shaped rewards stats - mean: %.5f, max: %.5f, min: %.5f",
            shaped_rewards.mean().item(), shaped_rewards.max().item(), shaped_rewards.min().item()
        )
    
        self.logger.info(
            "_shape_reward: Applied length bonus (max %.3f) and terminal rewards to batch of size %d",
            self.args.length_bonus_upper_bound, rewards.size(0)
        )
    
        return dict(
            shaped_rewards=shaped_rewards,
            non_score_rewards=non_score_rewards,
            kl=kl
        )

    


    
    def _estimate_advantage(
        self, rewards: torch.Tensor, values: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Generalized advantage estimation.

        Reference:
            https://arxiv.org/abs/1506.02438
        """
        # used args:

        # whiten_rewards.
        # gamma.
        # lam.
        # response_len.
        # whitening_async_stats


        # Mathematical formula : A(t) = δ_t + (γλ)δ_(t+1) + (γλ)²δ_(t+2) + ...
        # We calculate it recursively, starting by the last position.


        if self.args.whiten_rewards:
            
            print("Whitening");
            rewards = self.reward_normalizer.whiten(
                rewards, shift_mean=False
            )

        else:
            rewards = rewards * 10.0;
        

        lastgaelam = 0;

        advantages_reversed = []
        
        gen_length = self.args.response_len;
        
        for t in reversed(range(gen_length)):
            nextvalues = values[:, t + 1] if t < gen_length - 1 else 0.0
            delta = rewards[:, t] + self.args.gamma * nextvalues - values[:, t]
            lastgaelam = delta + self.args.gamma * self.args.lam * lastgaelam
            advantages_reversed.append(lastgaelam)
        
        advantages = torch.stack(advantages_reversed[::-1], dim=1)
        
        returns = advantages + values
        
        advantages = self.reward_normalizer.whiten(
            advantages, shift_mean=True
        )
        
        return dict(returns=returns, advantages=advantages)




        

    @torch.inference_mode()
    def rollout(self, queries_data) -> Dict[str, torch.Tensor]:
        """Rollout trajectories with policy.

        Args:
            queries_data: Sequence of batches or DataLoader.
                Each batch is a dict with keys 'queries' and 'query_attn_masks'.

        Returns:
            Dictionary with keys
                'queries', 'query_attn_masks', 'responses',
                'logprobs', 'ref_logprobs', 'values',
                'rewards', 'non_score_rewards', 'shaped_rewards'.
        """

        
        # used args: 
        # clean_tokens_after_eos

        
        self.policy.eval();

        
        self.ref_policy.eval();
        
        self.reward_model.eval();
        
        if self.second_reward_model:
            self.second_reward_model.eval();

        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu");      


        
        rollouts = [];
        
        for batch_idx, batch in tqdm.tqdm(
            enumerate(queries_data),
        
            total=len(queries_data),
            
            desc="rollout",
        ):
            
            gc.collect();

            
            
            torch.cuda.empty_cache();
            (
                pure_queries,
                queries,
                query_attn_masks,
                length_bonus_multiplier,
                question_types,
                structural_classes,
                ground_truth_outputs,
                
                
            ) = common_utils.unpack_dict(
                common_utils.prepare_inputs(batch, device=self.device),
                keys=(
                    "pure_queries",
                    "queries",
                    "query_attn_masks",
                    "length_bonus",
                    "question_types",
                    "structural_classes",
                    "ground_truth_outputs"
                    
                ),
            )




            if self.args.verifiable_reward:

                    unpadded_ground_truth = [];
                
                    for ground_truth_output in ground_truth_outputs:
                        
                        grt_mask = ground_truth_output != self.tokenizer.pad_token_id
                        
                        unpadded_grt = ground_truth_output[grt_mask]

                        unpadded_ground_truth.append(unpadded_grt);
                        
                    ground_truth_outputs_text = self.tokenizer.batch_decode(
                    unpadded_ground_truth,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                    );
                                
                    

                    

                
                
            
            respond_outputs = self.policy.respond(
                queries, query_attn_masks, temperature=self.args.temperature
            );

            
            
            (responses,) = common_utils.unpack_dict(respond_outputs, ("responses",))


            
            rollouts_batch = {
                "queries": queries,
                "query_attn_masks": query_attn_masks,
                "responses": responses,
            }
            
            
            unpadded_queries = [];
            
           
            for query in pure_queries:
    
                mask = query != self.tokenizer.pad_token_id
                unpadded_query = query[mask]
                unpadded_queries.append(unpadded_query)
            

            
            text_pure_queries = self.tokenizer.batch_decode(
                unpadded_queries,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False,
            );

        
            
            text_responses = self.tokenizer.batch_decode(
                
                responses,
                # skip_special_tokens=True,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            
            );
            # text_responses_verifiable = self.tokenizer.batch_decode(
                
            #     responses,
            #     skip_special_tokens=True,
            #     # skip_special_tokens=False,
            #     clean_up_tokenization_spaces=False,
            
            # );


            if isinstance(self.tokenizer, LlamaTokenizerFast):
                llama_fast_special_tokens = [self.tokenizer.eos_token]
                if self.tokenizer.eos_token != self.tokenizer.bos_token:
                    llama_fast_special_tokens.append(self.tokenizer.bos_token)
                    
                for special_token in llama_fast_special_tokens:
                    if special_token is not None:
                        text_responses = [
                            text_response.replace(special_token, f" {special_token}")
                            for text_response in text_responses
                        ]



            if self.args.data_config.stop_token is not None:
                parsed_stop_token = self.args.data_config.stop_token.replace(
                    r"\n", "\n"
                )
            else:
                parsed_stop_token = self.tokenizer.eos_token;

            
            text_responses = [
                truncate_after_stop_token(text_response, parsed_stop_token).split(
                    self.tokenizer.pad_token
                )[0]
                for text_response in text_responses
            ];

            
            text_responses_verifiable = [
                text_response.replace(self.tokenizer.eos_token, "")
                for text_response in text_responses
            ]


            verifiable_rewards = None;


            
            if self.args.verifiable_reward:
               
                assert len(text_responses_verifiable) == len(ground_truth_outputs_text);
                assert len(text_pure_queries) == len(ground_truth_outputs_text);
                
                verifiable_rewards = [];

                
                
                for i, reference_response in enumerate(ground_truth_outputs_text):
                    
                   print(f"++++++++++++++reference_response++++++++++++ {reference_response}");
                   print(f"+++++++++++++++++text response+++++++++++++ {text_responses_verifiable[i]}");
                   print(f"++++++++++++++text_pure_queries+++++++++++++++++ {text_pure_queries[i]}")
                   print(f"++++++++++++++structural_classes+++++++++++++++++ {structural_classes[i]}")
                   
                   structural_class = structural_classes[i].item()
                   print("Structural Class Text", idx_to_structural_class.get(structural_class, "Prose"))
                    
                                        
                   verifiable_reward_value = verifiable_reward(
                    text = text_responses_verifiable[i],
                    reference_text = reference_response,
                    input_context = text_pure_queries[i],
                    structural_class = idx_to_structural_class.get(structural_class,"Prose"));

                   print(f"========================The value of the verifiable reward===================== : {verifiable_reward_value}")
                   
                   verifiable_rewards.append(verifiable_reward_value)
                   
            
            
            

            has_stop_token = [
                parsed_stop_token in text_response for text_response in text_responses
            ];
            
            
            
            first_position_stop_token = [
                r.strip().startswith(self.tokenizer.eos_token)
                for r in text_responses
            ];


            

            del pure_queries, queries, responses  # Prevent mistakes.
        
            ## for now, we will use GENERAL(HELPFUL) = 0, redteam = 1, noAnswer = 2.
            question_type_flags = []
            
            for question_type in question_types.reshape(-1).tolist():
                if question_type == 1:
                    if self.args.enable_helpfulness_principles:
                        question_type_flags.append("REDTEAMING")    
                    else:
                        question_type_flags.append("HELPFUL")
                elif question_type == 2:
                    if self.args.enable_redteaming_principles:
                        question_type_flags.append("DONOTANSWER")
                    else:
                        question_type_flags.append("HELPFUL");   
                else:
                    question_type_flags.append("HELPFUL");



            text_sequences = [
                self.prepare_reward_inputs(
                    inputs=q, outputs=clean_after_stop_token(r), eos_token = self.tokenizer.eos_token ,question_type_flag=f
                )
                for q, r, f in common_utils.zip_(
                    text_pure_queries, text_responses, question_type_flags
                )   
            ];

            
            begin_padding_len = self.tokenizer(
                ["\n"], return_tensors="pt", add_special_tokens=False
            ).input_ids.shape[1]


            
            text_responses = ["\n" + r for r in text_responses];

            
            sequences = self.tokenizer(
                text_sequences, return_tensors="pt", padding=True, truncation=True
            )

            
            ori_padding_side = self.tokenizer.padding_side;
            
            
            self.tokenizer.padding_side = "right";
            
            
            responses = self.tokenizer(
                
                text_responses,
                
                return_tensors="pt",
                
                padding="max_length",
                
                truncation=True,
                
                max_length = self.args.response_len + begin_padding_len,
                
                add_special_tokens=False,
                
            );
            
            self.tokenizer.padding_side = ori_padding_side

            responses = responses.input_ids[:, begin_padding_len:]

            non_pad_mask = responses.ne(self.tokenizer.pad_token_id)
            
            non_pad_seq_len = (
                non_pad_mask.sum(dim=1).float().to(device)
            )
            
            length_bonus = non_pad_seq_len / float(self.args.response_len);


            length_bonus = length_bonus * length_bonus_multiplier.reshape(
                length_bonus.shape
            );

            self.logger.info(
            f"[length] mean={length_bonus.mean():.3f}, "
            f"min={length_bonus.min():.3f}, "
            f"max={length_bonus.max():.3f}"
        );
            


            sequences, responses = common_utils.prepare_inputs(
                (sequences, responses), device=device
            );


        

            if self.args.clean_tokens_after_eos:
                rollouts_batch["responses"] = responses


            batch_size_per_device = rollouts_batch["responses"].shape[0];


            sub_batch_size = self.args.reward_model_per_device_batch_size;

            
            if sub_batch_size is None or sub_batch_size == batch_size_per_device:

                ref_policy_outputs = self.ref_policy(
                    **rollouts_batch, temperature=self.args.temperature
                )

            
            else:
                assert batch_size_per_device % sub_batch_size == 0

                ref_policy_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    sub_batch = {
                        key: value[
                            sub_batch_idx
                            * sub_batch_size : (sub_batch_idx + 1)
                            * sub_batch_size
                        ]
                        for key, value in rollouts_batch.items()
                    }
                    sub_batch_ref_policy_outputs = self.ref_policy(
                        **sub_batch, temperature=self.args.temperature
                    )
                    ref_policy_outputs_list.append(sub_batch_ref_policy_outputs)

                ref_policy_outputs = common_utils.merge_dict(
                    ref_policy_outputs_list, merge_fn=torch.cat
                )
                
                del sub_batch_ref_policy_outputs
                del ref_policy_outputs_list
                del sub_batch



            if sub_batch_size is None or sub_batch_size == batch_size_per_device:
                policy_outputs = self.policy(
                    **rollouts_batch,
                    temperature=self.args.temperature,
                )
            else:
                assert batch_size_per_device % sub_batch_size == 0

                policy_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    sub_batch = {
                        key: value[
                            sub_batch_idx
                            * sub_batch_size : (sub_batch_idx + 1)
                            * sub_batch_size
                        ]
                        for key, value in rollouts_batch.items()
                    }
                    sub_batch_policy_outputs = self.policy(
                        **sub_batch,
                        temperature=self.args.temperature,
                    )
                    policy_outputs_list.append(sub_batch_policy_outputs)

                

                policy_outputs = common_utils.merge_dict(
                    policy_outputs_list, merge_fn=torch.cat
                );

                
                del sub_batch_policy_outputs
                del policy_outputs_list
                del sub_batch

                
            policy_outputs = common_utils.unpack_dict(
                policy_outputs,
                keys=("logprobs", "entropies", "values"),
                return_type=dict,
            );

            
            ref_policy_outputs = common_utils.unpack_dict(
                ref_policy_outputs, keys=("logprobs", "entropies"), return_type=dict
            )


            

            if torch.isnan(policy_outputs["logprobs"]).any():
                print("logprobs", policy_outputs["logprobs"][0])
            if torch.isnan(policy_outputs["values"]).any():
                print("values", policy_outputs["values"][0])
            if torch.isnan(ref_policy_outputs["logprobs"]).any():
                print("ref_logprobs", ref_policy_outputs["logprobs"][0])
            print("=" * 20)


            
            rollouts_batch.update(policy_outputs)
            
            rollouts_batch.update({f"ref_{k}": v for k, v in ref_policy_outputs.items()})
            
            rollouts_batch["length_bonus"] = length_bonus


            if sub_batch_size is None or sub_batch_size == batch_size_per_device:

                
                
                reward_outputs = self.reward_model(
                    input_ids=sequences.input_ids,
                    attention_mask=sequences.attention_mask,
                );

                print(f"Reward outputs shape is : {reward_outputs['rewards'].shape}")


                if self.second_reward_model:
                    second_reward_outputs = self.second_reward_model(
                        input_ids=sequences.input_ids,
                        attention_mask=sequences.attention_mask,
                    )
                    print("Second_reward", second_reward_outputs);
                    
                else:
                    second_reward_outputs = None;

                print("decoded reward input_ids", self.tokenizer.batch_decode(
                sequences.input_ids,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ));

                

            else:
                assert batch_size_per_device % sub_batch_size == 0

                reward_outputs_list = []

                for sub_batch_idx in range(batch_size_per_device // sub_batch_size):
                    idx_start = sub_batch_idx * sub_batch_size
                    idx_end = (sub_batch_idx + 1) * sub_batch_size
                    sub_batch_reward_outputs = self.reward_model(
                        input_ids=sequences.input_ids[idx_start:idx_end],
                        attention_mask=sequences.attention_mask[idx_start:idx_end],
                    )
                    reward_outputs_list.append(sub_batch_reward_outputs)

                reward_outputs = common_utils.merge_dict(
                    reward_outputs_list, merge_fn=torch.cat
                )
                del reward_outputs_list
                del sub_batch_reward_outputs
                
                
                print(f"Reward outputs{reward_outputs}")
                print("=" * 20);



            # print(f"\n\nThe rollout batch before post reward: {rollouts_batch} \n\n")

             
            reward_outputs = self.post_reward(
                reward_outputs,
                responses,
                penalize_no_stop_token=self.args.penalize_no_stop_token,
                relative_stop_token_penalty=self.args.relative_stop_token_penalty,
                has_stop_token=has_stop_token,
                second_reward_outputs = second_reward_outputs,
                verifiable_rewards = torch.tensor(verifiable_rewards, device = responses.device) if verifiable_rewards else None
                # first_position_stop_token = first_position_stop_token,
                # length_bonus = length_bonus,
            )





            
            print(f"----------------------Reward is {reward_outputs}-----------------------")

            rollouts_batch.update(reward_outputs)

            # print(f"\n\nThe rollout batch after post reward: {rollouts_batch} \n\n")

            # Shape reward with KL penalty.
            shape_reward_outputs = self._shape_reward(
                
                rewards=rollouts_batch["rewards"],
                responses=rollouts_batch["responses"],
                logprobs=rollouts_batch["logprobs"],
                ref_logprobs=rollouts_batch["ref_logprobs"],
                length_bonus=rollouts_batch["length_bonus"],
                
            );
        
            
            rollouts_batch.update(shape_reward_outputs);

            # print(f"\n\nThe rollout batch after reward shape: {rollouts_batch} \n\n")


            rollouts_batch_cpu = {
                key: value.cpu() for key, value in rollouts_batch.items()
            }
            
            rollouts.append(rollouts_batch_cpu);

        
        # Items in dict need to be of same shape.
        rollouts = common_utils.merge_dict(rollouts, merge_fn=torch.cat)


        
        # Estimating advantages outside the loop gives more samples for reward normalization.
        advantages = self._estimate_advantage(    
            rewards=rollouts["shaped_rewards"].to(device),
            values=rollouts["values"].to(device),
        )
        
        advantages = {key: value.cpu() for key, value in advantages.items()}
        
        return {**rollouts, **advantages};












    def log_scalars(self, rollouts_batch, prefix="rollout"):
        rewards = rollouts_batch["rewards"]
        shaped = rollouts_batch["shaped_rewards"]
        values = rollouts_batch["values"]
        lengths = rollouts_batch["output_lengths"]
    
        log = {
            f"{prefix}/reward_mean": rewards.mean().item(),
            f"{prefix}/reward_std": rewards.std().item(),
            f"{prefix}/shaped_reward_mean": shaped.mean().item(),
            f"{prefix}/value_mean": values.mean().item(),
            f"{prefix}/value_std": values.std().item(),
            f"{prefix}/output_len_mean": lengths.mean().item(),
            f"{prefix}/output_len_min": lengths.min().item(),
            f"{prefix}/output_len_max": lengths.max().item(),
        }
    
        for k, v in log.items():
            self.logger.info(f"{k}: {v:.4f}")
    



    def log_stop_tokens(self, has_stop_token, first_position_stop_token):
        has_stop = torch.tensor(has_stop_token).float()
        early_stop = torch.tensor(first_position_stop_token).float()
    
        self.logger.info(
            f"stop_token_rate={has_stop.mean().item():.3f}, "
            f"early_stop_rate={early_stop.mean().item():.3f}"
        )
    
  
    
    def post_reward(
        self,
        reward_outputs: Dict[str, torch.Tensor],
        responses: torch.Tensor,
        penalize_no_stop_token: bool,
        relative_stop_token_penalty: bool,
        has_stop_token: List[bool],
        second_reward_outputs: Dict[str, torch.Tensor] = None,
        verifiable_rewards: Dict[str, torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Assign bad reward values to sequences which didn't stop properly."""


        if verifiable_rewards:
            print(f"Using verifiable rewards {verifiable_rewards}")
            reward_outputs["rewards"] = verifiable_rewards;
        
        
        if second_reward_outputs:
            reward_outputs["rewards"] = 0.3*reward_outputs["rewards"] + 0.7*second_reward_outputs['rewards']


    
        if penalize_no_stop_token:
            has_stop_token = torch.tensor(has_stop_token, device=responses.device)  # shape: (B,)
        
            rewards = reward_outputs["rewards"]  # shape: (B,)
        
            eos_at_start = responses[:, 0] == 128001
            if relative_stop_token_penalty:

                # print("////Responses at first////",responses[:, 0:5]);
                      
                rewards[eos_at_start] = self.args.penalty_reward_value;
                
                missing_stop = ~has_stop_token;
        
                rewards = rewards + missing_stop.float() * self.args.penalty_reward_value
        
            else:
                rewards[~has_stop_token | eos_at_start] = self.args.penalty_reward_value
        
            reward_outputs["rewards"] = rewards
        
        return reward_outputs

            
    
        if self.args.truncate_token_ids is None:
            return reward_outputs

            
        
        def get_validity_mask(
            sequences: torch.Tensor, end_token_id: int
        ) -> torch.Tensor:
            """Mark a batch element as False if the sequence doesn't end with `end_token_id` after `truncate_after`."""
            assert sequences.dim() == 2
            validity_mask = []
            for sequence in sequences:
                (nonzeros,) = (sequence == end_token_id).nonzero(as_tuple=True)
                if len(nonzeros) == 0:
                    validity_mask.append(False)
                else:
                    validity_mask.append(
                        self.args.truncate_after is None
                        or
                        # Last occurrence of `end_token_id` is after `truncate_after`.
                        nonzeros[-1] > self.args.truncate_after
                    )
            return torch.tensor(validity_mask, device=sequences.device)
    
        validity_masks = [
            get_validity_mask(responses, end_token_id)
            for end_token_id in self.args.truncate_token_ids
        ]
        validity_mask = torch.stack(validity_masks).any(dim=0)
    
        rewards = reward_outputs["rewards"];
        
        rewards[~validity_mask] = self.args.penalty_reward_value;
        
        return reward_outputs
    
    









        
        


    
    def compute_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            values,
            old_logprob,
            returns,
            advantages,
            queries,
            query_attn_masks,
            responses,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "logprobs",
                    "returns",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                ),
            ),
            device=self.device,
        )

        # used args : temperature, 

        # Enable training mode for gradient checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries, query_attn_masks, responses, temperature=self.args.temperature
        );

        vpred = outputs["values"];

        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        )


        vf_losses1 = (vpred - returns) ** 2.0
        vf_losses2 = (vpredclipped - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()

        logprob = outputs["logprobs"];
        ratio = torch.exp(logprob - old_logprob);
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio;

        pg_losses2 = -advantages * torch.clamp(
            ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange
        )
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (
            (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()
        )  # noqa

        loss = pg_loss + self.args.vf_coef * vf_loss

        entropy = outputs["entropies"].mean()
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)
        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss, value=vf_loss, total=loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )


    

    def compute_policy_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        
        
        
        (
            values, ## Old value predictions.
            old_logprob, ##  old policy log probs
            returns, ## target returns.
            advantages, ## computed advantages A.
            queries, 
            query_attn_masks,
            responses,
        ) = common_utils.prepare_inputs(
            common_utils.unpack_dict(
                rollouts,
                keys=(
                    "values",
                    "logprobs",
                    "returns",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                ),
            ),
            device=self.device,
        )

        # Enable training mode for graident checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries,
            query_attn_masks,
            responses,
            temperature=self.args.temperature,
            mode="policy",
        )

        logprob = outputs["logprobs"]
        ratio = torch.exp(logprob - old_logprob)
        # When current policy is close to the old policy, the KL component of this advantage is approximately correct.
        pg_losses = -advantages * ratio
        pg_losses2 = -advantages * torch.clamp(
            ratio, min=1.0 - self.args.cliprange, max=1.0 + self.args.cliprange
        )
        pg_loss = torch.maximum(pg_losses, pg_losses2).mean()
        pg_clipfrac = (
            (pg_losses2 > pg_losses).to(torch.get_default_dtype()).mean()
        )  # noqa


        entropy = outputs["entropies"].mean()

        beta_entropy = 0.001  # small positive coefficient
        # loss = pg_loss + outputs["dummy_loss"] - beta_entropy * entropy
        loss = pg_loss + outputs["dummy_loss"];


        
        approxkl = 0.5 * ((logprob - old_logprob) ** 2.0).mean()

        return_mean, return_var = returns.mean(), returns.var(unbiased=False)

        stats = dict(
            loss=dict(policy=pg_loss),
            policy=dict(entropy=entropy, approxkl=approxkl, clipfrac=pg_clipfrac),
            returns=dict(mean=return_mean, var=return_var),
        )
        
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )


    



    def compute_value_loss(
        self, rollouts: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict]:
        (
            values,
            old_logprob,
            returns,
            advantages,
            queries,
            query_attn_masks,
            responses,
        ) = common_utils.prepare_inputs(
            
            common_utils.unpack_dict(
                
                rollouts,
                
                keys=(
                    "values",
                    "logprobs",
                    "returns",
                    "advantages",
                    "queries",
                    "query_attn_masks",
                    "responses",
                ),

            ),

            device=self.device,
        
        )

        # Enable training mode for graident checkpointing.
        self.policy.train()

        outputs = self.policy(
            queries,
            query_attn_masks,
            responses,
            temperature=self.args.temperature,
            mode="value",
        )

        vpred = outputs["values"];
        
        vpredclipped = torch.clamp(
            vpred,
            min=values - self.args.cliprange_value,
            max=values + self.args.cliprange_value,
        )
        
        vf_losses1 = (vpred - returns) ** 2.0
        vf_losses2 = (vpredclipped - returns) ** 2.0
        vf_loss = 0.5 * torch.maximum(vf_losses1, vf_losses2).mean()
        vf_clipfrac = (vf_losses2 > vf_losses1).to(torch.get_default_dtype()).mean()

        loss = self.args.vf_coef * vf_loss + outputs["dummy_loss"]

        value_mean, value_var = values.mean(), values.var(unbiased=False)

        stats = dict(
            loss=dict(value=vf_loss),
            val=dict(
                vpred=vpred.mean(),
                error=((vpred - returns) ** 2).mean(),
                clipfrac=vf_clipfrac,
                mean=value_mean,
                var=value_var,
            ),
        )
        
        return loss, common_utils.flatten_dict(
            stats, sep="/", postprocess_fn=lambda x: x.detach()
        )


    

    ### TODO record_step_stats

    ### It should be tested :

    
    

    def save_model(
        self,
        output_dir: Optional[str] = None,
        give_rw_access=True,
        check_corrupted=True,
    ):


        os.makedirs(output_dir, exist_ok=True);
        
        print("Saving model checkpoint to %s" % output_dir);
        
        peft_model_path = os.path.join(output_dir, ADAPTER_MODEL_DIR);

        
        ### to test

        # print("self.policy : ", self.policy);

        
        policy_model = self.policy.policy;


        save_adapters(
                policy_model.base_model,
                peft_model_path,
                # adapter_names=["lora_policy","lora_default"],
                adapter_names=["lora_policy"],
        )

        
        pytorch_model_paths = glob.glob(
                os.path.join(output_dir, "pytorch_model*.bin")
            );
        

        
        
        for pytorch_model_path in pytorch_model_paths:
            
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path);


        
        value_model = self.policy.value_model;

        
        save_adapters(
            value_model.base_model,
            peft_model_path,
            adapter_names=["lora_value"],
        )

        
        torch.save(
            value_model.value_head.state_dict(),
            os.path.join(output_dir, VALUE_HEAD_NAME),
        )
        

        pytorch_model_paths = glob.glob(
            os.path.join(output_dir, "pytorch_model*.bin")
        );


        for pytorch_model_path in pytorch_model_paths:
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)

        torch.save(
                self.optimizer.state_dict(), os.path.join(output_dir, OPTIMIZER_NAME)
            )
            # Save scheduler.
        
        torch.save(
                self.lr_scheduler.state_dict(), os.path.join(output_dir, SCHEDULER_NAME)
            )

            # Delete other optimizer checkpoints to save disk space.
            # glob pattern to match all optimizer.pt files in the father directory
        pattern = os.path.join(os.path.dirname(output_dir), "*/optimizer.pt")

            # get a list of all matching paths
        optimizer_files = glob.glob(pattern)

            # iterate over the optimizer files
        for file in optimizer_files:
                # if the file is not in the output_dir, delete it
                if output_dir not in file:
                    os.remove(file)


                    





            
            
    def record_step_stats(self, train_stats, rollouts, step_idx, **kwargs):
        """
        Record statistics for a single training step, including PPO metrics, KL, rewards, entropies, and length bonus.
    
        Args:
            train_stats (dict): PPO update statistics (loss, advantages, etc.)
            rollouts (dict): Rollout outputs from policy evaluation
            step_idx (int): Current training step
            kwargs: Additional info, e.g., kl_coef
        Returns:
            stats (dict): Flattened dictionary of logged metrics
        """
    
        kl = rollouts["kl"]
        kl_sum_seq = kl.sum(dim=1).mean(dim=0)
        kl_avg_seq = kl.mean()
        
        shaped_rewards = rollouts["shaped_rewards"].sum(dim=1).mean(dim=0)
        non_score_rewards = rollouts["non_score_rewards"].sum(dim=1).mean(dim=0)
        rewards = rollouts["rewards"].mean(dim=0)
        length_bonus = rollouts.get("length_bonus", torch.tensor(0.0)).mean()
        entropies = rollouts.get("entropies", torch.tensor(0.0)).mean()
        ref_entropies = rollouts.get("ref_entropies", torch.tensor(0.0)).mean()
        lr = self.optimizer.param_groups[0]["lr"]
    
        stats = {
            "objective/kl_coef": kwargs.get("kl_coef", 0.0),
            "objective/kl_sum_seq": kl_sum_seq,
            "objective/kl_avg_seq": kl_avg_seq,
            "objective/length_bonus": length_bonus,
            "objective/shaped_rewards": shaped_rewards,
            "objective/non_score_rewards": non_score_rewards,
            "objective/rewards": rewards,
            "objective/lr": lr,
            "objective/entropies": entropies,
            "objective/ref_entropies": ref_entropies,
        }
    
        # Add PPO train stats
        for k, v in train_stats.items():
            stats[f"ppo/{k}"] = v.mean(dim=0)
    
        # Convert all tensors to Python scalars
        stats = {
            key: value.item() if torch.is_tensor(value) else value
            for key, value in stats.items()
        }

        os.makedirs(self.args.output_dir, exist_ok=True)
        stats_path = os.path.join(self.args.output_dir, "training_stats.jsonl")
        with open(stats_path, "a", encoding="utf-8") as sf:
            json.dump(stats, sf)
            sf.write("\n")
    
        # --- Logging ---
        self.logger.info(
            "Step %d | KL coef: %.5f | KL sum: %.5f | KL avg: %.5f | Length bonus: %.5f | "
            "Shaped rewards: %.5f | Non-score rewards: %.5f | Rewards: %.5f | LR: %.6f | Entropy: %.5f",
            step_idx, stats["objective/kl_coef"], stats["objective/kl_sum_seq"],
            stats["objective/kl_avg_seq"], stats["objective/length_bonus"],
            stats["objective/shaped_rewards"], stats["objective/non_score_rewards"],
            stats["objective/rewards"], stats["objective/lr"], stats["objective/entropies"]
        )
    
        self.logger.debug(
            "Step %d | Entropies: %.5f | Ref entropies: %.5f | PPO stats: %s",
            step_idx, stats["objective/entropies"], stats["objective/ref_entropies"],
            {k: v for k, v in stats.items() if k.startswith("ppo/")}
        )

        
    
        # Optional: log rollout texts if needed
        if getattr(self.args, "log_rollouts_text", False):
            queries_text = self.tokenizer.batch_decode(
                rollouts["queries"], skip_special_tokens=True
            )
            responses_text = self.tokenizer.batch_decode(
                rollouts["responses"], skip_special_tokens=True
            )
            self.logger.debug("Step %d | Sample queries: %s", step_idx, queries_text[:3])
            self.logger.debug("Step %d | Sample responses: %s", step_idx, responses_text[:3])
    
        return stats



    
    # totest:
    @torch.inference_mode()
    def resume_training(self, checkpoint_dir):
        # Load optimizer.
        ### Training checkpoint for example is saved here
        # checkpoints/
        #   └── checkpoint-4000/
        #         ├── optimizer.pt
        #         ├── scheduler.pt

        # Load optimizer state
        
        optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
        
        if os.path.exists(optimizer_path) and self.args.load_optimizer:
            self.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
            print(f"Optimizer state loaded from {optimizer_path}")
        else:
            print(f"A brand new optimizer is loaded")
        # Load scheduler state
        scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
        
        if os.path.exists(scheduler_path) and self.args.load_optimizer:
            self.lr_scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))
            print(f"Scheduler state loaded from {scheduler_path}")
        else:
            print(f"\\\\\\/////\\\\\ A brand new LR scheduler is loaded\\\\\//////\\//")

        # Extract the step number from the checkpoint name (e.g., "checkpoint-4000")
        match = re.search(r"checkpoint-(\d+)", checkpoint_dir)
        if match:
            skipping_steps = int(match.group(1))
        else:
            skipping_steps = 0  # fallback if no step info in name

        print(f"Resuming training from step {skipping_steps}")
        return skipping_steps


    
    def smart_tokenizer_and_embedding_resize(
        num_new_tokens: int,
        tokenizer: transformers.PreTrainedTokenizer,
        model: transformers.PreTrainedModel,
    ):
        if num_new_tokens > 0:
            model.resize_token_embeddings(len(tokenizer))
            input_embeddings = model.get_input_embeddings().weight.data
            output_embeddings = model.get_output_embeddings().weight.data

            input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )
            output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                dim=0, keepdim=True
            )

            input_embeddings[-num_new_tokens:] = input_embeddings_avg
            output_embeddings[-num_new_tokens:] = output_embeddings_avg

            model.get_input_embeddings().requires_grad_(True)
            model.get_output_embeddings().requires_grad_(True)






# from models.reward_model_GMM import load_4bit_gmm_reward_model_for_inference





        
def make_models(
    tokenizer: transformers.PreTrainedTokenizer,
    reward_tokenizer: transformers.PreTrainedTokenizer,
    args,
    num_new_tokens: int = 0,
    reward_num_new_tokens: int = 0,
    resume_from_checkpoint: Optional[str] = None,
    ) -> dict:

    
    print(f"----------------------------Resume from checkpoint from make_models : {resume_from_checkpoint}------------------------------");

    
    def make_generative_policy(
        adapter_name,
        is_trainable,
        reuse_base_model=True,
        resume_path=None,
        fully_initialize=False,
        policy_model_bits=4,
    ):


        
        print(f"Making policy model with adapter -> {adapter_name}, is_trainable -> {is_trainable}, resume_path ->{resume_path} , fully_initialize {fully_initialize}");
        
        model = load_4bit_model_for_inference(
            checkpoint_dir = resume_path,
            bits=  policy_model_bits,
            fp16 = args.fp16,
            bf16 = args.bf16,
            gradient_checkpointing = args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            # reuse_base_model=reuse_base_model,
            # reuse_base_model=False,
            reuse_base_model=False,
            trust_remote_code = args.trust_remote_code,
            base_model_mapping = args.base_model_mapping,
            fully_initialize = fully_initialize,
            base_model_name_or_path_for_fully_initialize = args.base_model_name_or_path_for_fully_initialize,
        );
        
        try:
            if hasattr(model, "peft_config"):
                for name, cfg in model.peft_config.items():
                    tgt = getattr(cfg, "target_modules", None)
                    # print(f"[DEBUG-MODEL] make_generative_policy(adapter={adapter_name}, is_trainable={is_trainable}) -> adapter '{name}': peft_type={getattr(cfg,'peft_type',None)}, target_modules={tgt}")
            else:
                print(f"[DEBUG-MODEL] make_generative_policy: model has no 'peft_config' attribute")
        except Exception as e:
            print(f"[DEBUG-MODEL] Error inspecting model.peft_config: {e}")
        

        if args.bf16 and not isinstance(model, LlamaForCausalLM):
            model = model.to(torch.bfloat16)

        return model



        
    def make_reward_model(
  adapter_name, is_trainable, reuse_base_model=True, resume_path=args.reward_model_name_or_path, use_gmm = False
    
    ):

        print(f"Making reward model with adapter -> {adapter_name}, is_trainable -> {is_trainable}, resume_path ->{resume_path}");


            
        model = load_4bit_reward_model_for_inference(
            # checkpoint_dir=resume_path or args.reward_model_name_or_path,
            checkpoint_dir = resume_path,
            # checkpoint_dir = args.reward_model_name_or_path,
            bits=args.reward_model_bits,
            fp16=args.fp16,
            bf16=args.bf16,
            gradient_checkpointing=args.gradient_checkpointing,
            adapter_name=adapter_name,
            is_trainable=is_trainable,
            reuse_base_model=reuse_base_model,
            # reuse_base_model=False,
            trust_remote_code=args.trust_remote_code,
            base_model_mapping=args.base_model_mapping,
        )


        
        # Debug: inspect reward model peft_config
        try:
            if hasattr(model, "peft_config"):
                for name, cfg in model.peft_config.items():
                    tgt = getattr(cfg, "target_modules", None)
                    print(f"[DEBUG-MODEL] make_reward_model(adapter={adapter_name}, is_trainable={is_trainable}) -> adapter '{name}': peft_type={getattr(cfg,'peft_type',None)}, target_modules={tgt}")
            elif hasattr(model, "backbone_model") and hasattr(model.backbone_model, "peft_config"):
                for name, cfg in model.backbone_model.peft_config.items():
                    tgt = getattr(cfg, "target_modules", None)
                    print(f"[DEBUG-MODEL] make_reward_model.backbone(adapter={adapter_name}) -> adapter '{name}': peft_type={getattr(cfg,'peft_type',None)}, target_modules={tgt}")
            else:
                print(f"[DEBUG-MODEL] make_reward_model: no peft_config found on model or backbone_model")
        except Exception as e:
            print(f"[DEBUG-MODEL] Error inspecting reward model peft_config: {e}")

        
        if args.bf16 and not isinstance(model.backbone_model, LlamaForCausalLM):
            model = model.to(torch.bfloat16)
        
        return model



    policy_resume_path = None


    
    if resume_from_checkpoint:
        policy_resume_path = os.path.join(
        resume_from_checkpoint, ADAPTER_MODEL_DIR, "lora_policy"
        )



                


    policy = rl_models.make_policy_with_base_model(
        args,
        make_generative_policy(
            adapter_name="lora_policy",
            is_trainable=True,
            resume_path=policy_resume_path,
            fully_initialize=args.fully_initialize_policy,
        ),
        tokenizer,
        adapter_name="lora_policy",
    );

    print("++++++++++++++++++++++++++++++++++++++====================\n\n")
    
    print(f"\n\n\n---------------------REFERENCE policy resumed from checkpoint{args.reference_model_dir}\n\n\n\n\n-----------------");
    
    ref_policy = rl_models.make_policy_with_base_model(
        
        args,
        
        make_generative_policy(
            adapter_name= "lora_policy",
            is_trainable=False,
            resume_path = args.reference_model_dir if args.reference_model_dir else None,    
            fully_initialize = True,
        ),
        
        tokenizer,
        adapter_name="lora_policy",
    
    )
    
    

    print(f"\n\n\n\n --------------------- Policy resumed from checkpoint{policy_resume_path} ----------------- \n\n\n\n");


    

    value_resume_path = None;
    
    value_head_resume_path = None;

    
    if resume_from_checkpoint:
        
        value_resume_path = os.path.join(
            resume_from_checkpoint, ADAPTER_MODEL_DIR, "lora_value"
        )
        
        value_head_resume_path = os.path.join(resume_from_checkpoint, VALUE_HEAD_NAME)
        print(f"---------------------Value model resumed from checkpoint {value_head_resume_path}\n\n\n\n\n\n-----------------");



    reward_model = make_reward_model(
        adapter_name="lora_reward",
        is_trainable=False,
        resume_path = args.reward_model_name_or_path,
        use_gmm = args.use_gmm,
    );

    
    if args.init_value_with_reward:
        # Initialize value from reward model a la OAI.
        print("Initializing value model with reward model.");
        
        value_model = rl_models.make_value_with_base_model(
            args,
            make_reward_model(
                adapter_name="lora_value",
                is_trainable=True,
                resume_path=args.reward_model_name_or_path,
                use_gmm = args.use_gmm,
            ).backbone_model,

            reward_tokenizer,
            adapter_name="lora_value",
        );
        
    else:
        print("Initializing value model with policy model.")
        # Initialize value from policy. Works for sanity, but generally performs worse in instruction-following.
        value_model = rl_models.make_value_with_base_model(
            args,
            make_generative_policy(
                adapter_name="lora_value",
                is_trainable=True,
                # reuse_base_model=False,
                # policy_model_bits=4,
                resume_path=value_resume_path,
                fully_initialize=args.fully_initialize_policy,
            ),
            tokenizer,
            adapter_name="lora_value",
        )
        

    if value_head_resume_path:
        value_model.value_head.load_state_dict(
            torch.load(value_head_resume_path, map_location="cpu")
        )

    
    actor_critic = rl_models.ActorCritic(policy=policy, value_model=value_model)


    # second_reward_model = make_reward_model(
    #     adapter_name="lora_reward",
    #     is_trainable=False,
    #     resume_path = args.second_reward_model_name_or_path,
    #     use_gmm = args.use_gmm,
    # )
    second_reward_model = None;


    if num_new_tokens > 0:
        smart_tokenizer_and_embedding_resize(
            num_new_tokens, tokenizer, policy.base_model
        );
        
    if reward_num_new_tokens > 0:
        smart_tokenizer_and_embedding_resize(
            reward_num_new_tokens, reward_tokenizer, reward_model.backbone_model
        )
    # ref_policy.requires_grad_(False)
    # # ref_policy = accelerator.prepare(ref_policy)  # noqa
    # reward_model.requires_grad_(False)
    # if not args.init_value_with_reward:
    #     reward_model = accelerator.prepare(reward_model)
    # # TODO: This is a hack to get FSDP running. Remove in the future when we figure things out.
    # if accelerator.distributed_type == accelerate.DistributedType.FSDP:
    #     inputs = tokenizer("fsdp are you happy now??? :)" * 50, return_tensors="pt")
    #     inputs = {key: value.to(accelerator.device) for key, value in inputs.items()}
    #     actor_critic(inputs["input_ids"], inputs["attention_mask"], inputs["input_ids"])
    return dict(policy=actor_critic, ref_policy=ref_policy, reward_model=reward_model, second_reward_model = second_reward_model);                       







class RewardNormalizer:
    def __init__(self, epsilon=1e-8):
        self.epsilon = epsilon

    def whiten(self, values: torch.Tensor, shift_mean=True):
        
        if values.size(0) < 2:
            return values

        else:
            mean = values.mean(dim=-1, keepdim=True)
            std = values.std(dim=-1, unbiased=False, keepdim=True)

            whitened = (values - mean) / (std + self.epsilon)

            if not shift_mean:
                whitened = whitened + mean
            return whitened






    







def clean_after_stop_token(
    response: str,
    additional_stop_token: Optional[List[str]] = None,
) -> str:

    
    if additional_stop_token is None:
        additional_stop_token = ["<s>", "</s>", "###","##", "<|endoftext|>"]

    for token in additional_stop_token:
        response = response.split(token)[0]

    return response.strip()
    # return response













def truncate_after_stop_token(
    response: str,
    stop_token: Optional[str] = None,
    additional_stop_token: Optional[List[str]] = None,
) -> str:
    if stop_token is None:
        return response

    if additional_stop_token is None:
        additional_stop_token = [
            "<s>",
            "</s>",
            "\nUpdate",
            "\nNote", 
            "\nComment",
            "##",
            "\nExplanation",
            "Explanation",
            "Note",
        ]

    for token in additional_stop_token + [stop_token]:
        if len(response.split(token)) > 1:
            response = response.split(token)[0] + token

    
    if "###" in response:
        response = (
            response.split("###")[0]
            + "###"
            + " ".join(response.split("###")[1].split(" ")[:2])
        )

    return response
    # return response.strip()







    
def save_adapters(model, save_directory, adapter_names, **kwargs):

    
    r"""
    This function saves the adapter model and the adapter configuration files to a directory, so that it can be
    reloaded using the [`LoraModel.from_pretrained`] class method, and also used by the [`LoraModel.push_to_hub`]
    method.

    Args:
        model: The model to save.
        save_directory (`str`):
            Directory where the adapter model and configuration files will be saved (will be created if it does not
            exist).
        adapter_name (`str`):
            Name of the adapter to save.
        kwargs (additional keyword arguments, *optional*):
            Additional keyword arguments passed along to the `push_to_hub` method.
    """

    
    if os.path.isfile(save_directory):
        
        raise ValueError(
            f"Provided path ({save_directory}) should be a directory, not a file"
        )
        
    os.makedirs(save_directory, exist_ok=True)

    for adapter_name, peft_config in model.peft_config.items():

        if adapter_name in adapter_names:
            # save only the trainable weights
            # Debug: print config target modules and basic info to help diagnose mismatches
            # try:
            #     tgt = getattr(peft_config, "target_modules", None)
            #     # print(f"[DEBUG] Adapter '{adapter_name}': peft_type={getattr(peft_config, 'peft_type', None)}, target_modules={tgt}")
            #     if isinstance(tgt, (list, tuple)):
            #         print(f"[DEBUG] Adapter '{adapter_name}' target_modules length: {len(tgt)}")
            # except Exception as e:
            #     print(f"[DEBUG] Could not inspect peft_config for adapter '{adapter_name}': {e}")

            # save only the trainable weights
            
        
            
            output_state_dict = get_peft_model_state_dict(
                model,
                state_dict=kwargs.get("state_dict", None),
                adapter_name=adapter_name,
            );


            
            output_dir = (
                os.path.join(save_directory, adapter_name)
                if adapter_name != "default"
                else save_directory
            );



            
            os.makedirs(output_dir, exist_ok=True)

            
            torch.save(output_state_dict, os.path.join(output_dir, WEIGHTS_NAME))

            # save the config and change the inference mode to `True`

            
            if peft_config.base_model_name_or_path is None:
                
                peft_config.base_model_name_or_path = (
                    model.base_model.model.__dict__.get("name_or_path", None)
                )


                
                
            inference_mode = peft_config.inference_mode
            
            peft_config.inference_mode = True
            
            peft_config.save_pretrained(output_dir)
            
            peft_config.inference_mode = inference_mode




                          

