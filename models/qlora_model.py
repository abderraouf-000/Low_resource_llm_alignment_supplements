
import os
from argparse import Namespace
from typing import Optional
from os.path import join, exists
from peft.tuners.lora import LoraLayer

import torch
import bitsandbytes as bnb
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)

from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    PeftModel,
    PeftModelForCausalLM,
)

from transformers import LlamaForCausalLM


REGISTERED_BASE_MODELS = {}


def find_all_linear_names(
    args: Namespace,
    model: torch.nn.Module,
):
    cls = (
        bnb.nn.Linear4bit
        if args.bits == 4
        else (bnb.nn.Linear8bitLt if args.bits == 8 else torch.nn.Linear)
    )
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            if "lora" not in names[-1]:
                lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if "lm_head" in lora_module_names:  # needed for 16-bit
        lora_module_names.remove("lm_head")
    return list(lora_module_names)






def get_accelerate_model(
    args: Namespace,
    checkpoint_dir: Optional[str] = None,
    adapter_name= "lora_default",
    is_trainable=True,
    reuse_base_model=False,
):

    
    global REGISTERED_BASE_MODELS

    compute_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )


    ## this is true in our case e.g : model_saving
    if checkpoint_dir is not None:
        if os.path.exists(os.path.join(checkpoint_dir, "adapter_model")):
            checkpoint_dir = os.path.join(checkpoint_dir, "adapter_model")


        ### this is true in our case model_saving/lora_default.
        
        if os.path.exists(os.path.join(checkpoint_dir, "lora_default")):
            
            checkpoint_dir = os.path.join(checkpoint_dir, "lora_default")

            

    ## in our case is false :
    
    if args.model_name_or_path in REGISTERED_BASE_MODELS and reuse_base_model:
        
        print("OUR model is already registered and reuse_base_model == True");
        
        config = {
            
            "load_in_4bit": args.bits == 4,
            "load_in_8bit": args.bits == 8,
            "llm_int8_threshold": 6.0,
            "llm_int8_has_fp16_weight": False,
            "bnb_4bit_compute_dtype": compute_dtype,
            "bnb_4bit_use_double_quant": args.double_quant,
            "bnb_4bit_quant_type": args.quant_type,
        }

        


        
        registered_model, registered_config = REGISTERED_BASE_MODELS[
            args.model_name_or_path
        ]

        
        
        if registered_config == config and not args.full_finetune:
            
            print(f"loading registered model {args.model_name_or_path}...")
            model = registered_model

            if checkpoint_dir is not None:
                print(f"Check point dir {checkpoint_dir} is not NONE, Adapter {adapter_name} and trainable is set to {is_trainable}")
                model.load_adapter(
                    checkpoint_dir,
                    adapter_name=adapter_name,
                    is_trainable=is_trainable,
                )
                
            else:
                modules = args.lora_modules or find_all_linear_names(args, model);
                print("adding LoRa modules: ", modules);
                
                config = LoraConfig(
                    r=args.lora_r,
                    lora_alpha=args.lora_alpha,
                    target_modules=modules,
                    lora_dropout=args.lora_dropout,
                    bias="none",
                    task_type="CAUSAL_LM",
                )
                
                model.add_adapter(adapter_name, peft_config=config)
            return model
        else:
            raise ValueError(
                f"Model {args.model_name_or_path} is already registered with a different config."
                f"{registered_config} != {config}"
            )

    current_device = torch.cuda.current_device()



    
    if args.full_finetune:
        assert args.bits in [16, 32]

    print(f"loading base model {args.model_name_or_path}...")

    CausalLM = AutoModelForCausalLM;

    if "falcon" in args.model_name_or_path.lower():
        CausalLM = RWForCausalLM
    elif (
        "llama" in args.model_name_or_path.lower()
        or "vicuna" in args.model_name_or_path.lower()
        or "dromedary" in args.model_name_or_path.lower()
    ) and torch.__version__ >= "2.0.0":
        CausalLM = LlamaForCausalLM

    model = CausalLM.from_pretrained(
        args.model_name_or_path,
        # load_in_4bit=args.bits == 4,
        # load_in_8bit=args.bits == 8,
        device_map={"": current_device},
        # max_memory=max_memory,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=args.bits == 4,
            load_in_8bit=args.bits == 8,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=args.double_quant,
            bnb_4bit_quant_type=args.quant_type,
        ),
        torch_dtype=(
            torch.float16
            if args.fp16
            else (torch.bfloat16 if args.bf16 else torch.float32)
        ),
        trust_remote_code=args.trust_remote_code,
    )
    
    if compute_dtype == torch.float16 and args.bits == 4:
        major, minor = torch.cuda.get_device_capability()
        if major >= 8:
            print("=" * 80)
            print(
                "Your GPU supports bfloat16, you can accelerate training with the argument --bf16"
            )
            print("=" * 80)

    setattr(model, "model_parallel", True)
    setattr(model, "is_parallelizable", True)

    model.config.torch_dtype = (
        torch.float16 if args.fp16 else (torch.bfloat16 if args.bf16 else torch.float32)
    )

    ### True in our case 
    
    if not args.full_finetune:
        
        model = prepare_model_for_kbit_training(
            model, use_gradient_checkpointing=args.gradient_checkpointing
        )
        
        ## True in our case.
        
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    
    ### True in our case
    
    if not args.full_finetune:

        
        if checkpoint_dir is not None:

            print("+++++++++++++++++++++++++++++++++++++\n")
            print(f"Loading adapters from checkpoint :{checkpoint_dir} .\n");
            print("+++++++++++++++++++++++++++++++++++++")

            
            print(f"the arguments for adapters are : \n \n adapter_name : {args}")
            
            
            model = PeftModel.from_pretrained(
                model,
                checkpoint_dir,
                adapter_name=adapter_name,
                is_trainable=is_trainable,
            );
            
            
        else:
            
            print(f'adding NEW LoRA modules...');
            
            modules = args.lora_modules or find_all_linear_names(args, model)
            
            print("adding LoRa modules: ", modules)
            
            config = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                target_modules=modules,
                lora_dropout=args.lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )

            print(f"the peftConfig is : {config}")
            model = get_peft_model(model, config, adapter_name=adapter_name)

        ## true in our first case
        if args.model_name_or_path not in REGISTERED_BASE_MODELS:
            config = {
                "load_in_4bit": args.bits == 4,
                "load_in_8bit": args.bits == 8,
                "llm_int8_threshold": 6.0,
                "llm_int8_has_fp16_weight": False,
                "bnb_4bit_compute_dtype": compute_dtype,
                "bnb_4bit_use_double_quant": args.double_quant,
                "bnb_4bit_quant_type": args.quant_type,
            }
            print("Adding to registered");
            REGISTERED_BASE_MODELS[args.model_name_or_path] = (model, config)

    for name, module in model.named_modules():
        if isinstance(module, LoraLayer):
            if args.bf16:
                module = module.to(torch.bfloat16)
            else:
                module = module.to(torch.float32)
        if "lm_head" in name or "embed_tokens" in name:
            if hasattr(module, "weight"):
                if args.bf16 and module.weight.dtype == torch.float32:
                    module = module.to(torch.bfloat16)
                # if not args.bf16:
                #     module = module.to(torch.float32)
    return model





def load_4bit_model_for_inference(
    checkpoint_dir: str,
    bits: int = 4,
    fp16: bool = False,
    bf16: bool = False,
    double_quant: bool = True,
    quant_type: str = "nf4",
    gradient_checkpointing: bool = False,
    adapter_name="lora_default",
    is_trainable=True,
    reuse_base_model=False,
    trust_remote_code=False,
    base_model_mapping=None,
    fully_initialize=False,
    base_model_name_or_path_for_fully_initialize :str = None,
):


    
    print(f"checkpoint directory : {checkpoint_dir} \n");
    
    if checkpoint_dir is not None and base_model_name_or_path_for_fully_initialize is None:
        
        if exists(join(checkpoint_dir, "adapter_model")):
            checkpoint_dir = join(checkpoint_dir, "adapter_model")

        if exists(join(checkpoint_dir, "lora_default")):
            checkpoint_dir = join(checkpoint_dir, "lora_default")

        config = LoraConfig.from_pretrained(checkpoint_dir)
        base_model_name_or_path = config.base_model_name_or_path

    elif base_model_name_or_path_for_fully_initialize is not None:
        base_model_name_or_path = base_model_name_or_path_for_fully_initialize;

    
    if base_model_mapping is not None:
        dict_base_model_mapping = eval(base_model_mapping)
        if (
            dict_base_model_mapping is not None
            and base_model_name_or_path in dict_base_model_mapping
        ):
            base_model_name_or_path = dict_base_model_mapping[base_model_name_or_path]

    
    args = Namespace(
        model_name_or_path=base_model_name_or_path,
        bits=bits,
        fp16=fp16,
        bf16=bf16,
        double_quant=double_quant,
        quant_type=quant_type,
        gradient_checkpointing=gradient_checkpointing,
        trust_remote_code=trust_remote_code,
        full_finetune=False,
        lora_r = 32 if fully_initialize else None,
        lora_alpha = 16 if fully_initialize else None,
        lora_dropout = 0.0 if fully_initialize else None,
        lora_modules = None,
    )


    print(f"Current QloRA model args : {args}")

    
    if fully_initialize:
        print("Fully initializing qlora model.")

    
    model = get_accelerate_model(
        args,
        checkpoint_dir= None if fully_initialize else checkpoint_dir,
        adapter_name = adapter_name,
        is_trainable=is_trainable,
        reuse_base_model=reuse_base_model,
    )
    
    return model

    


def get_peft_model(model, peft_config, adapter_name="default"):
    """
    Returns a Peft model object from a model and a config.

    Args:
        model ([`transformers.PreTrainedModel`]): Model to be wrapped.
        peft_config ([`PeftConfig`]): Configuration object containing the parameters of the Peft model.
    """
    peft_config.base_model_name_or_path = model.__dict__.get("name_or_path", None)
    return PeftModelForCausalLM(model, peft_config, adapter_name=adapter_name)




    