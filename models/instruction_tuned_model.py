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
    adapter_name="lora_Instruction",
    is_trainable=True,
    reuse_base_model=True,
    **kwargs,
):
    model_cls = transformers.LlamaForCausalLM

    if qlora:
        if checkpoint_dir is None or checkpoint_dir in ["scratch", "none"]:
            
            return get_accelerate_model(args, adapter_name = adapter_name);
        
        else:
            print("= == === = = == = = = = Adapter name is ",adapter_name)
            
            return get_accelerate_model(
                args,
                checkpoint_dir=checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
                reuse_base_model=reuse_base_model,
            );
            
    return model_cls.from_pretrained(model_name_or_path, **kwargs)








class InstructionConfig(transformers.PretrainedConfig):
    model_type = "Instruction_model"

    # Huggingface doesn't allow non-kwargs for `__init__`.
    def __init__(self, backbone_model_name_or_path=None, **kwargs):
        super(InstructionConfig, self).__init__(**kwargs)
        self.backbone_model_name_or_path = backbone_model_name_or_path






@dataclass
class InstructionModelOutput(ModelOutput):
    Instructions: Tensor = None






    
class InstructionModel(transformers.PreTrainedModel):
    
    config_class = InstructionConfig;
    
    supports_gradient_checkpointing = True;

    def __init__(
        self,
        args: Namespace,
        config: InstructionConfig,
        checkpoint_dir: Optional[str] = None,
        adapter_name = "lora_default",
        **kwargs,
    ):
        
        super(InstructionModel, self).__init__(config);
        
        self.adapter_name = adapter_name;

        print("=== === = === == = == = = = = =The adapter name ", adapter_name)
        
        self.backbone_model = make_generative_lm(
            args,
            config.backbone_model_name_or_path,
            checkpoint_dir=checkpoint_dir,
            adapter_name=adapter_name,
            **kwargs,
        );
        

        

    def forward(self, input_ids, attention_mask=None, return_dict=True, **kwargs):
        
        self.backbone_model.set_adapter(self.adapter_name);
        
        self.backbone_model.config.use_cache = False;

        outputs = self.backbone_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=return_dict,
            output_hidden_states=False,
            **kwargs,
        );

        return outputs

        

    def get_input_embeddings(self):
        """Return the input embeddings from the base model"""
        return self.backbone_model.get_input_embeddings()
    
    def set_input_embeddings(self, value):
        """Set the input embeddings for the base model"""
        self.backbone_model.set_input_embeddings(value)
        

    
    def save_pretrained(self, save_directory: str, **kwargs):
        """
        Save LoRA adapters to the specified directory.
        """
        # Ensure the directory exists
        os.makedirs(save_directory, exist_ok=True)
    
        # Save LoRA adapters
        self.backbone_model.save_pretrained(save_directory, **kwargs)
    








        
def load_4bit_Instruction_model_for_inference(
    checkpoint_dir: str,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=True,
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

    config = InstructionConfig(backbone_model_name_or_path=base_model_name_or_path)

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

    model = InstructionModel(
        args,
        config,
        checkpoint_dir=checkpoint_dir,
        qlora=bits == 4 or bits == 8,
        adapter_name=adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    
    return model