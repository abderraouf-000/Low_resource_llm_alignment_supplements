from dataclasses import dataclass
from argparse import Namespace
import os
from typing import Optional, Dict, Sequence, Union

import einops
import torch
from torch import Tensor, nn
import torch.nn.functional as F

import transformers
from transformers.trainer_utils import EvalPrediction
from transformers.utils.generic import ModelOutput

from peft import PeftModel, LoraModel, LoraConfig

from models.qlora_model import get_accelerate_model

from transformers import LlamaForCausalLM






def unpack_dict(
    d: Dict, keys: Sequence[str], return_type: type = tuple
) -> Union[Sequence, Dict]:
    if return_type in (tuple, list):
        return return_type(d[key] for key in keys)
    elif return_type == dict:
        return {key: d[key] for key in keys}
    else:
        raise ValueError(f"Unknown return_type: {return_type}")





def batch_select(input: Tensor, index: Tensor):
    """Select elements from a batched tensor with a batched index tensor.

    Example:
        input = torch.tensor([
            [0, 1, 2],
            [3, 0, 9],
            [6, 7, 8],
        ])
        index = torch.tensor([[0, 1], [1, 0], [0, 0]])
        batch_select(input, index) = tensor([
            [0, 1],
            [0, 3],
            [6, 6]
        ])
    """
    dummy_index = torch.arange(input.size(0), device=input.device).unsqueeze(-1)
    return input[dummy_index, index]





def make_generative_lm(
    args: Namespace,
    model_name_or_path: str,
    qlora: bool = False,
    checkpoint_dir: Optional[str] = None,
    adapter_name="lora_reward",
    is_trainable=True,
    reuse_base_model=False,
    **kwargs,
):
    
    model_cls = transformers.LlamaForCausalLM

    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            
            return get_accelerate_model(args, None);
        
        else:
            ## this is true in our case
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                reuse_base_model=reuse_base_model,
            );
            
    return model_cls.from_pretrained(model_name_or_path, **kwargs)


def get_transformer_hidden_size(model: transformers.PreTrainedModel):
    if isinstance(model, PeftModel):
        return get_transformer_hidden_size(model.base_model)

    if isinstance(model, LoraModel):
        return get_transformer_hidden_size(model.model)

    if isinstance(model, transformers.GPT2LMHeadModel):
        hidden_size_attr_name = "n_embd"
    elif isinstance(model, transformers.OPTForCausalLM):
        hidden_size_attr_name = "word_embed_proj_dim"
    elif isinstance(model, transformers.T5ForConditionalGeneration):
        hidden_size_attr_name = "d_model"
    elif isinstance(model, transformers.GPTBigCodeForCausalLM):
        hidden_size_attr_name = "n_embd"
    elif "modelling_RW.RWModel" in str(
        type(model)
    ) or "modelling_RW.RWForCausalLM" in str(type(model)):
        hidden_size_attr_name = "hidden_size"
    else:
        llama_cls = getattr(
            transformers,
            "LLaMAForCausalLM"
            if hasattr(transformers, "LLaMAForCausalLM")
            else "LlamaForCausalLM",
        )
        if isinstance(model, llama_cls) or "LlamaForCausalLM" in str(type(model)):
            hidden_size_attr_name = "hidden_size"
        else:
            raise ValueError(f"Unknown base_model type: {type(model)}")
        from typing import Any, Mapping
    return getattr(model.config, hidden_size_attr_name)

    




































        
        
        
        












class RewardConfig(transformers.PretrainedConfig):
    model_type = "reward_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(RewardConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path

    
@dataclass
class RewardModelOutput(ModelOutput):
    rewards: Tensor = None
    # embeddings: Tensor = None 
    hidden : Tensor = None



    
class RewardModel(transformers.PreTrainedModel):
    
    config_class = RewardConfig;
    
    supports_gradient_checkpointing = True;

    def __init__(
        self,
        args: Namespace,
        config: RewardConfig,
        checkpoint_dir: Optional[str] = None,
        adapter_name = "lora_default",
        **kwargs,
    ):
        
        super(RewardModel, self).__init__(config);

        
        self.adapter_name = adapter_name;

        
        self.backbone_model = make_generative_lm(
            args,
            config.backbone_model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            adapter_name=adapter_name,
            **kwargs,
        );
        

        hidden_size = get_transformer_hidden_size(self.backbone_model)

        reward_head = nn.Linear(hidden_size, 1);
                
        torch.nn.init.zeros_(reward_head.bias);
        
        device = next(self.backbone_model.parameters()).device
        
        self.reward_head = reward_head.to(device);        

        if checkpoint_dir is not None:
            reward_head_path = os.path.join(checkpoint_dir, "reward_head");
            
            if os.path.exists(reward_head_path):
                self.reward_head.load_state_dict(
                    torch.load(
                        reward_head_path,
                        map_location="cpu",
                    )
                )
            else:
                print(f"Warning: reward head not found at {reward_head_path}")

        
        is_trainable = kwargs.get("is_trainable", True)
        self.reward_head.requires_grad_(is_trainable)
        
        
    
    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        # This function only computes rewards (no loss), useful for reranking or RL stages.
        
        # Ensure we use the correct LoRA adapter
        self.backbone_model.set_adapter(self.adapter_name)
        self.backbone_model.config.use_cache = False
    
        # Run the backbone transformer
        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
            output_hidden_states=True,
            **kwargs,
        )
    
        last_hidden_state = outputs.hidden_states[-1]
        assert isinstance(last_hidden_state, torch.Tensor), f"{outputs}"
    
        # Ensure all parameters contribute to gradient flow
        logits = outputs.logits
        last_hidden_state = last_hidden_state + 0.0 * torch.mean(logits)

        
        # Take the final token representation (end of sequence)
        last_hidden_state_at_the_end = last_hidden_state[:, -1, :]

        
    
        # Match dtype/device with reward head parameters
        final_dtype_tensor = next(self.reward_head.parameters())
        last_hidden_state_at_the_end = last_hidden_state_at_the_end.type_as(final_dtype_tensor)
    
        if final_dtype_tensor.dtype == torch.float16:
            last_hidden_state_at_the_end = last_hidden_state_at_the_end.to(torch.float32)
    
        # Pass through intermediate projection and reward head
        hidden =  last_hidden_state_at_the_end
        
        rewards = self.reward_head(hidden).squeeze(-1);

        # print("Rewards",rewards);
    
        # Return structured or tuple output
        if return_dict:
            return RewardModelOutput(
                rewards=rewards,
                hidden = hidden,
            )
        else:
            # return rewards, last_hidden_state_at_the_end, hidden
            return rewards, hidden



        

    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save the entire model (base model + LoRA adapters + intermediate + reward head) 
        to the specified directory.
        """
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)
    
        # Save the backbone model with LoRA adapters
        self.backbone_model.save_pretrained(save_directory, **kwargs)
    

        # Save the reward head
        reward_head_path = os.path.join(save_directory, "reward_head")
        torch.save(self.reward_head.state_dict(), reward_head_path)
        print(f"Reward head saved to {reward_head_path}")
    





        

def compute_pairwise_loss_with_smoothing(
    rewards_0,
    rewards_1,
    choice,
    smoothing: float = 0.05,
    return_outputs: bool = False,
):
    """
    Compute binary cross-entropy loss on pairwise rewards with label smoothing.

    Args:
        rewards_0 (torch.Tensor): Rewards for first response (shape: [B]).
        rewards_1 (torch.Tensor): Rewards for second response (shape: [B]).
        choice (torch.Tensor): Binary label (1 if response_1 > response_0 else 0).
        smoothing (float): Label smoothing factor in [0, 0.5).
        return_outputs (bool): Whether to return logits dict.

    Returns:
        loss (torch.Tensor) or (loss, {"logits": logits})
    """
    # Compute logits = reward difference
    logits = rewards_1 - rewards_0  # shape: (B,)

    # Apply label smoothing
    smoothed_labels = choice.to(logits.dtype) * (1 - smoothing) + (smoothing / 2)

    # Compute binary cross-entropy with logits
    loss = F.binary_cross_entropy_with_logits(
        logits, smoothed_labels, reduction="mean"
    )

    if return_outputs:
        return loss, {"logits": logits, "labels": smoothed_labels}
        
    return loss,logits




    

class RewardModelTrainer(transformers.Trainer):




    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # input_ids, attention_mask each of size (bsz, num_candidates, seq_len).
        # index_0, index_1 each of size (bsz, num_pairs); indexes into input_ids.
        # choice of size (bsz, num_pairs); 1 if index_1's seq is chosen, 0 otherwise.
        
        input_ids, attention_mask, index_0, index_1, choice = unpack_dict(
            inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
        )
        
        num_candidates, num_pairs = input_ids.size(1), choice.size(1)
        input_ids_flat, attention_mask_flat = tuple(
            einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
        )
        outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
        rewards_flat = outputs.rewards
        rewards = einops.rearrange(
            rewards_flat, "(b c) -> b c", c=num_candidates
        )  # Size: (bsz, num_candidates).

        
        rewards_0, rewards_1 = tuple(
            batch_select(rewards, index) for index in (index_0, index_1)
        )  # Size: (bsz, num_pairs).
        
        loss,logits = compute_pairwise_loss_with_smoothing(
            rewards_0,
            rewards_1,
            choice,
            smoothing = 0.2,
            return_outputs = False,
        );
        
        # logits = rewards_1 - rewards_0  # Size: (bsz, num_pairs).

        # loss = F.binary_cross_entropy_with_logits(
        #     logits, choice.to(logits.dtype), reduction="mean"
        # )

        return (loss, dict(logits=logits)) if return_outputs else loss

        

        # probs = torch.sigmoid(logits);
        
        # # Type casting of `choice` is due to amp.autocast context manager.
        
        # target = choice.to(logits.dtype);
        
        # alpha = 0.4;
        
        # loss = - (  (1 - alpha) * target * torch.log(probs + 1e-12) + 
        #           alpha * (1 - target) * torch.log(1 - probs + 1e-12)  );
        
        # loss = loss.mean()

    
        
    # def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None, preference_strengths=None):
    #     """
    #     Compute LSAM loss for a batch of preference pairs.
    
    #     Args:
    #         model: reward model that outputs scalar rewards for sequences.
    #         inputs: dict containing
    #             - input_ids: (bsz, num_candidates, seq_len)
    #             - attention_mask: (bsz, num_candidates, seq_len)
    #             - index_0: (bsz, num_pairs), indices of first candidate
    #             - index_1: (bsz, num_pairs), indices of second candidate
    #             - choice: (bsz, num_pairs), 1 if index_1 is preferred, else 0
    #         preference_strengths: optional tensor (bsz, num_pairs) for adaptive α
    #             If None, defaults to fixed α=0.6
    #         return_outputs: whether to return model outputs
    #     """
    
    #     # Unpack inputs
    #     input_ids, attention_mask, index_0, index_1, choice = unpack_dict(
    #         inputs, keys=("input_ids", "attention_mask", "index_0", "index_1", "choice")
    #     )
    #     bsz, num_candidates, seq_len = input_ids.size()
    #     num_pairs = choice.size(1)
    
    #     # Flatten candidates for model forward pass
    #     input_ids_flat, attention_mask_flat = tuple(
    #         einops.rearrange(x, "b c l -> (b c) l") for x in (input_ids, attention_mask)
    #     )
    
    #     outputs = model(input_ids=input_ids_flat, attention_mask=attention_mask_flat)
    #     rewards_flat = outputs.rewards  # shape: (bsz * num_candidates, 1) or (bsz * num_candidates,)
        
    #     # Reshape back to (bsz, num_candidates)
    #     rewards = einops.rearrange(rewards_flat, "(b c) -> b c", c=num_candidates)
    
    #     # Select rewards for the preference pairs
    #     rewards_0, rewards_1 = tuple(
    #         batch_select(rewards, index) for index in (index_0, index_1)
    #     )  # shape: (bsz, num_pairs)
    
    #     # Compute logits and predicted probabilities
    #     logits = rewards_1 - rewards_0
    #     probs = torch.sigmoid(logits)
    
    #     # Cast target to match dtype (for AMP)
    #     target = choice.to(logits.dtype)
    
    #     # Compute adaptive alpha
    #     if preference_strengths is not None:
    #         adaptive_alpha = torch.sigmoid(preference_strengths)  # shape: (bsz, num_pairs)
    #     else:
    #         adaptive_alpha = torch.full_like(target, 0.4)  # fallback fixed α
    
    #     # LSAM loss
    #     loss = - (adaptive_alpha * target * torch.log(probs + 1e-12) +
    #               (1 - adaptive_alpha) * (1 - target) * torch.log(1 - probs + 1e-12))
    #     loss = loss.mean()

        
    #     # loss = F.binary_cross_entropy_with_logits(
    #     #     logits, choice.to(logits.dtype), reduction="mean"
    #     # )
        
    #     return (loss, dict(logits=logits)) if return_outputs else loss
    
            


def compute_reward_modeling_metrics(eval_prediction: EvalPrediction) -> Dict:
    # eval_prediction.label_ids is a tuple that matches up with `training_args.label_names`.
    logits = torch.tensor(eval_prediction.predictions).squeeze(-1)
    labels = torch.tensor(eval_prediction.label_ids[-1]).squeeze(-1)
    predictions = (logits >= 0.0).long()
    accuracy = predictions.eq(labels).float().mean().item()
    label_positive_rate = (labels == 1).float().mean().item()
    positive_rate = (predictions == 1).float().mean().item()
    true_positive_rate = (predictions * labels).float().sum().item() / labels.sum().item()
    false_positive_rate = (predictions * (1 - labels)).float().sum().item() / (1 - labels).sum().item()
    return dict(
        accuracy=accuracy,
        label_positive_rate=label_positive_rate,
        positive_rate=positive_rate,
        true_positive_rate=true_positive_rate,
        false_positive_rate=false_positive_rate,
    )









        
def load_4bit_reward_model_for_inference(
    checkpoint_dir: str,
    bits: int = 4,
    fp16: bool = True,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
    base_model_mapping=None,
):

    
    
    # Load the model.
    lora_checkpoint_dir = checkpoint_dir;
    
    if os.path.exists(os.path.join(lora_checkpoint_dir, "adapter_model")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "adapter_model");

        
    if os.path.exists(os.path.join(lora_checkpoint_dir, "lora_default")):
        lora_checkpoint_dir = os.path.join(lora_checkpoint_dir, "lora_default");

    lora_config = LoraConfig.from_pretrained(lora_checkpoint_dir)
    base_model_name_or_path = lora_config.base_model_name_or_path

    if base_model_mapping is not None:
        dict_base_model_mapping = eval(base_model_mapping)
        if (
            dict_base_model_mapping is not None
            and base_model_name_or_path in dict_base_model_mapping
        ):
            base_model_name_or_path = dict_base_model_mapping[base_model_name_or_path]

    config = RewardConfig(backbone_model_name_or_path=base_model_name_or_path)

    args = Namespace(
        model_name_or_path=config.backbone_model_name_or_path,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        gradient_checkpointing=gradient_checkpointing,
    )

    model = RewardModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=bits == 4 or bits == 8,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    
    return model