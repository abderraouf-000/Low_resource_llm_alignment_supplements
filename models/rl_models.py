import abc
import logging
from typing import Dict, Optional

import torch
import transformers
from torch import Tensor, nn

from data_utils.common_utils import right_pad, compute_logprobs
from models.reward_model import get_transformer_hidden_size



class Policy(nn.Module, abc.ABC):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        self.adapter_name = adapter_name

    @abc.abstractmethod
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        assert not self.training, "Policy must be in eval model for generation."
        return self._post_respond(
            self._respond(queries, query_attn_masks, temperature, num_return_sequences)
        )

    @abc.abstractmethod
    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
        raise NotImplementedError

    def _post_respond(self, respond_outputs: Dict[str, Tensor]) -> Dict[str, Tensor]:
        return respond_outputs


class AutoregressivePolicy(Policy):
    
    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:

        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
            
        self.base_model.config.use_cache = False

        if temperature is None:
            temperature = self.args.temperature;
            
        input_ids = torch.cat([queries, responses], dim=1)
        
        attention_mask = input_ids.ne(self.base_tokenizer.pad_token_id)
        
        attention_mask[:, : queries.size(1)] = query_attn_masks
                
        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )

        outputs = self.base_model(**inputs, output_hidden_states=True);
        
        original_logits = outputs.logits[:, -self.args.response_len - 1 : -1];
        
        logits = original_logits / temperature;
        
        labels = input_ids[:, -self.args.response_len :];
        
        logprobs = compute_logprobs(
            logits, labels, ignore_index=self.base_tokenizer.pad_token_id
        );
        
        entropies = -(logits.softmax(dim=-1) * logits.log_softmax(dim=-1)).sum(dim=-1);

        
        last_hidden_state = outputs.hidden_states[-1][
            :, -self.args.response_len - 1 : -1
        ]
        return dict(
            original_logits=original_logits,
            logits=logits,
            logprobs=logprobs,
            entropies=entropies,
            last_hidden_state=last_hidden_state,
        )


    
    def _respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
        num_return_sequences=1,
    ) -> Dict[str, Tensor]:
    
        
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
            
        self.base_model.config.use_cache = True
        self.base_model.config.cache_shape = (
            queries.shape[-1] + self.args.response_len,
        )

        
        if temperature is None:
            temperature = self.args.temperature

            
            
        sequences = self.base_model.generate(
            
            inputs=queries,
            attention_mask=query_attn_masks,
            do_sample=True,
            max_new_tokens=self.args.response_len,
            pad_token_id=self.base_tokenizer.pad_token_id,
            eos_token_id=self.base_tokenizer.eos_token_id,
            top_p=1,
            top_k=0,
            temperature=temperature,
            num_return_sequences=num_return_sequences,
            synced_gpus=False,
            
        )
        



        # input_lengths = query_attn_masks.sum(dim=1)  # attention mask sums: input tokens count

        
# Slice out the generated part
        responses = sequences[:, queries.size(1):]  # assumes same padding for batch, adjust if variable



        
        return dict(
            responses=responses
        )  # Size (bsz * num_return_sequences, response_len).




class Value(nn.Module, abc.ABC):
    def __init__(
        self,
        args,
        base_model: transformers.PreTrainedModel,
        base_tokenizer: transformers.PreTrainedTokenizer,
        adapter_name: Optional[str] = None,
    ):
        super().__init__()
        self.args = args
        self.base_model = base_model
        self.base_tokenizer = base_tokenizer
        hidden_size = get_transformer_hidden_size(base_model)
        value_head = torch.nn.Linear(hidden_size, 1)
        value_head.weight.data.zero_()
        value_head.bias.data.zero_()
        self.value_head = value_head.to(next(base_model.parameters()).device)
        self.adapter_name = adapter_name

    @abc.abstractmethod
    def forward(
        self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor
    ) -> Dict[str, Tensor]:
        raise NotImplementedError


class AutoregressiveValue(Value):
    def forward(
        self, queries: Tensor, query_attn_masks: Tensor, responses: Tensor
    ) -> Dict[str, Tensor]:
        if self.adapter_name is not None:
            self.base_model.set_adapter(self.adapter_name)
        self.base_model.config.use_cache = False

        sequences = torch.cat([queries, responses], dim=1)
        sequence_attn_masks = sequences.ne(self.base_tokenizer.pad_token_id)

        inputs = self.base_model.prepare_inputs_for_generation(
            input_ids=sequences,
            attention_mask=sequence_attn_masks,
            use_cache=False,
        )
        outputs = self.base_model(
            **inputs,
            return_dict=True,
            output_hidden_states=True,
        )
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)
        last_hidden_state = last_hidden_state[:, queries.size(1) - 1 : -1]
        last_hidden_state = last_hidden_state.type_as(
            next(self.value_head.parameters())
        )
        values = self.value_head(last_hidden_state).squeeze(-1)
        return dict(values=values)



    
class ActorCritic(nn.Module):
    def __init__(self, policy: Policy, value_model: Value):
        super(ActorCritic, self).__init__()
        self.policy = policy
        self.value_model = value_model

    def forward(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        responses: Tensor,
        temperature: Optional[float] = None,
        mode: Optional[str] = None,
    ) -> Dict[str, Tensor]:
        # Assume the policy and value model share the same tokenizer.

        if mode is None:
            o1 = self.policy(queries, query_attn_masks, responses, temperature)
            o2 = self.value_model(queries, query_attn_masks, responses)

        elif mode == "policy":
            o1 = self.policy(queries, query_attn_masks, responses, temperature)
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o2 = {
                "dummy_loss": 0.0
                * torch.sum(
                    torch.stack(
                        [
                            torch.mean(value)
                            for key, value in self.named_parameters()
                            if "lora_value" in key
                        ]
                    )
                )
            }
        elif mode == "value":
            o2 = self.value_model(queries, query_attn_masks, responses)
            # Add dummy loss to make sure every parameter is used in the backward pass.
            o1 = {
                "dummy_loss": 0.0
                * torch.sum(
                    torch.stack(
                        [
                            torch.mean(value)
                            for key, value in self.named_parameters()
                            if "lora_policy" in key
                        ]
                    )
                )
            }
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return {**o1, **o2}

    def respond(
        self,
        queries: Tensor,
        query_attn_masks: Tensor,
        temperature: Optional[float] = None,
    ) -> Dict[str, Tensor]:
        return self.policy.respond(
            queries=queries, query_attn_masks=query_attn_masks, temperature=temperature
        )



def make_policy_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    adapter_name: Optional[str] = "default",
) -> Policy:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressivePolicy(
            args, base_model, base_tokenizer, adapter_name=adapter_name
        )


def make_value_with_base_model(
    args,
    base_model: transformers.PreTrainedModel,
    base_tokenizer: transformers.PreTrainedTokenizer,
    adapter_name: Optional[str] = "default",
) -> Value:
    if base_model.config.is_encoder_decoder:
        raise NotImplementedError
    else:
        return AutoregressiveValue(
            args, base_model, base_tokenizer, adapter_name=adapter_name
        )