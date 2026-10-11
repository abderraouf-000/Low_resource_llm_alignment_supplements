from argparse import Namespace
from datasets import load_dataset
import transformers
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
import tqdm
import json
import random
import pandas as pd
from peft import LoraConfig, PeftConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from huggingface_hub import login
import fire
import os
from collections import defaultdict
from typing import Optional
from models.qlora_model import load_4bit_model_for_inference


from models.reward_model import (
    RewardConfig,
    RewardModel,
)



BASE_PROMPT_DICT = {
    "prompt_input": "{instruction}\n\n{input}",
    "prompt_no_input": "{instruction}",
}



DEFAULT_META_PROMPT = """You are a reviewer whose goal is to judge the quality of the AI system's responses to instructions.

Your task is to evaluate the quality of the response. There are several dimensions you should consider in your evaluation:

- The AI should be tailored to the nature of the user query, taking into account who is interacting with the AI, as well as the situational context in which the assistant is being engaged.

A good response should meet all of the above criteria.


User: {Input}

response: {Output}

The quality of the response is"""



class RewardModelEvaluator:
    def __init__(self, model, tokenizer, meta_prompt: Optional[str] = None,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize the evaluator with a reward model and tokenizer.

        Args:
            model: The reward model
            tokenizer: The tokenizer for the model
            meta_prompt: The meta prompt template used during training
            device: Device to run inference on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.meta_prompt = meta_prompt if meta_prompt is not None else DEFAULT_META_PROMPT
        self.device = device
        self.model.to(device)
        self.model.eval()


    def format_prompt(self, instruction: str, instruction_input: str, output: str) -> str:
        """
        Format the prompt exactly as done during training.

        Args:
            instruction: The instruction text
            instruction_input: Optional input/context text
            output: The output/response text

        Returns:
            Formatted prompt string
        """
        if instruction_input:
            prompt_format = BASE_PROMPT_DICT["prompt_input"]
            formatted_input = prompt_format.format(instruction=instruction, input=instruction_input)
        else:
            prompt_format = BASE_PROMPT_DICT["prompt_no_input"]
            formatted_input = prompt_format.format(instruction=instruction)

        formatted_prompt = self.meta_prompt.format(
            Input=formatted_input,
            Output=output,
        ) + self.tokenizer.eos_token

        return formatted_prompt


    def get_reward_score(self, instruction: str, instruction_input: str, output: str) -> float:
        """
        Get reward score for a given instruction-output pair.

        Args:
            instruction: instruction
            instruction_input: input
            output: The model output to score

        Returns:
            Reward score as a float
        """
        text = self.format_prompt(instruction, instruction_input, output)

        with torch.no_grad():
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                padding='longest',
                max_length=self.tokenizer.model_max_length,
                truncation=True
            )
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            reward = outputs['rewards'].squeeze().cpu().item()
            return reward



HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)


prompt_template = """
Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.

### Instruction:
{Input}

### Response:
"""


def generate_result(model, tokenizer, prompts, max_new_tokens=128, greedy = True):

    if type(prompts) == str:
        prompts = [prompts]

    bad_words = ['\\', '\\\\', '`', '```']
    bad_words_ids = tokenizer(bad_words, add_special_tokens=False).input_ids

    all_results = [None for _ in range(len(prompts))]

    with torch.no_grad():
        inputs = tokenizer(prompts, return_tensors='pt', padding=True).to('cuda')

        if greedy:
            results = model.generate(**inputs, temperature=0., do_sample=False, max_new_tokens=max_new_tokens)
        else:
            results = model.generate(**inputs, temperature=0.7, do_sample=True, top_p=1, top_k=0, max_new_tokens=max_new_tokens)

        for i in range(len(prompts)):
            result = results[i][len(inputs['input_ids'][i]):]
            result = tokenizer.decode(result, skip_special_tokens=True)
            all_results[i] = result
    return all_results


def generate_with_retry(model, tokenizer, prompts, max_new_tokens=128, max_retries=3, greedy = True):
    """
    Generate responses and retry individually for any empty outputs.
    Returns a list of responses aligned with the input prompts.
    """
    responses = generate_result(model, tokenizer, prompts, max_new_tokens, greedy = greedy)

    for i, (prompt, response) in enumerate(zip(prompts, responses)):
        if not response or not response.strip():
            print(f"  [Retry] Empty response for prompt index {i}, retrying up to {max_retries} times...")
            for attempt in range(max_retries):
                retry_response = generate_result(model, tokenizer, [prompt], max_new_tokens, greedy = greedy)
                if retry_response and retry_response[0].strip():
                    responses[i] = retry_response[0]
                    print(f"  [Retry] Got non-empty response on attempt {attempt + 1}")
                    break
                else:
                    print(f"  [Retry] Attempt {attempt + 1} still empty.")
            else:
                print(f"  [Retry] All {max_retries} retries exhausted for prompt index {i}. Keeping empty string.")
                responses[i] = ""

    return responses


def build_prompts(batch, dataset_type):
    """Build templated prompts for a batch of items."""
    prompts = []
    for item in batch:
        if dataset_type in ["humaneval", "humaneval_instruct"]:
            templated_prompt = item['prompt']
        else:
            if item['input']:
                prompt = BASE_PROMPT_DICT["prompt_input"].format(
                    instruction=item['instruction'],
                    input=item['input']
                )
            else:
                prompt = BASE_PROMPT_DICT["prompt_no_input"].format(
                    instruction=item['instruction']
                )
            templated_prompt = prompt_template.format(Input=prompt)
        prompts.append(templated_prompt)
    return prompts


def build_output_record(item, response, dataset_type, model_string, gen_idx, bon_score=None):
    """Build a single output record dict."""
    instruction = item.get('instruction', item.get('prompt', ''))
    instruction_input = item.get('input', '')

    if dataset_type in ["humaneval", "humaneval_instruct"]:
        record = {
            'task_id': item['task_id'],
            'completion': response,
            'generation_index': gen_idx,
        }
    elif dataset_type == "gsm8k":
        record = {
            'instruction': instruction,
            'input': instruction_input,
            'answer': item.get('answer', ''),
            'output': response,
            'generator': model_string,
            'generation_index': gen_idx,
        }

    elif dataset_type == "IFEval":
        record = {
            'key': item['key'],
            'prompt': item['prompt'],
            'response': response,
        }
        
        
    else:
        record = {
            'instruction': instruction,
            'input': instruction_input,
            'output': response,
            'generator': model_string,
            'generation_index': gen_idx,
        }

    if bon_score is not None:
        record['bon_score'] = bon_score

    return record


def best_of_n_sampling(
    model,
    tokenizer,
    reward_evaluator: RewardModelEvaluator,
    items: list,
    dataset_type: str,
    n: int,
    max_retries: int,
    response_len: int,
    model_string: str,
    gen_idx: int,
    greedy: bool,
) -> list:
    """
    For each item in the batch, generate N candidates and pick the one
    with the highest reward score.

    Args:
        model: The generation model
        tokenizer: The generation tokenizer
        reward_evaluator: RewardModelEvaluator instance for scoring
        items: List of instruction dicts for the current batch
        dataset_type: Type of dataset being evaluated
        n: Number of candidates to generate per prompt
        max_retries: Max retries for empty responses
        response_len: Max new tokens for generation
        model_string: Model identifier string for output metadata
        gen_idx: Current generation index

    Returns:
        List of output dicts with the best response selected per item
    """
    prompts = build_prompts(items, dataset_type)
    

    # Generate N candidates per prompt — shape: all_candidates[n_idx][prompt_idx]
    all_candidates = []
    for n_idx in range(n):
        print(f"    [BoN] Generating candidate {n_idx + 1}/{n}...")
        
        candidates = generate_with_retry(
            model, tokenizer, prompts,
            max_retries=max_retries,
            max_new_tokens=response_len,
            greedy = greedy,
        )
        all_candidates.append(candidates)

    # Score all N candidates per item and pick the best
    batch_outputs = []
    for j, item in enumerate(items):
        instruction = item.get('instruction', item.get('prompt', ''))
        instruction_input = item.get('input', '')

        best_response = None
        best_score = float('-inf')

        for n_idx in range(n):
            response = all_candidates[n_idx][j]
            if not response or not response.strip():
                continue

            score = reward_evaluator.get_reward_score(
                instruction=instruction,
                instruction_input=instruction_input,
                output=response,
            )
            print(f"    [BoN] Item {j}, candidate {n_idx + 1}: score={score:.4f}")

            if score > best_score:
                best_score = score
                best_response = response

        if best_response is None:
            best_response = ""
            best_score = None
            print(f"    [BoN] Item {j}: all candidates empty, using empty string.")

        batch_outputs.append(build_output_record(
            item, best_response, dataset_type, model_string, gen_idx, bon_score=best_score
        ))

    return batch_outputs


from datasets import load_dataset


def main(
    use_base_model: bool = True,
    instruction_field: str = "instruction",
    Human_eval_file: str = "Evaluations/Eval_data/human-eval-v2-20210705.jsonl",
    Alpaca_eval_file: str = "Evaluations/Eval_data/alpaca_eval.json",
    IFEval_eval_file: str = "Evaluations/Eval_data/IF_Eval.jsonl",
    batchSize: int = 28,
    outFile: str = "Evaluations/Generations/Human_eval_completion_generation_base_greedy_len128.jsonl",
    modelString: str = "meta-llama/Llama-3.2-3B",
    dataset_type: str = "humaneval",
    num_generations: int = 1,
    max_retries: int = 8,
    response_len: int = 256,
    checkpoint_dir: str = "",
    reward_model_checkpoint_dir: str= "",
    greedy: bool = True,
    # checkpoint_dir: str = "PPO/Llama3.2-3b-ppo-rulebased-level3_checkpoint350_prose&nonprose/ppo-checkpoint-250/adapter_model/lora_policy",
    bon_n: int = 1,  # Set to 1 to disable BoN, >1 to enable (e.g. --bon_n 4)
):


    print(dataset_type);
    if dataset_type == "humaneval":
        evalFile = Human_eval_file
    elif dataset_type == "alpaca":
        evalFile = Alpaca_eval_file
    elif dataset_type == "IFEval":
        evalFile = IFEval_eval_file
    else:
        evalFile = None
    
        

    args = Namespace(
        use_base_model=use_base_model,
        instructionField=instruction_field,
        evalFile=evalFile,
        batchSize=batchSize,
        modelString=modelString,
        outFile=outFile,
        datasetType=dataset_type,
        numGenerations=num_generations,
        maxRetries=max_retries,
        response_len=response_len,
        checkpoint_dir=checkpoint_dir,
        bon_n=bon_n,
        reward_model_checkpoint_dir = reward_model_checkpoint_dir,
        greedy = greedy,
    )

    use_bon = args.bon_n > 1  # Derived flag — no separate argument needed
    
    if use_bon:
        args.greedy = False;
        
    print(f"Best-of-N sampling: {'ENABLED (N=' + str(args.bon_n) + ')' if use_bon else 'DISABLED'}")

    ## Loading evaluation set
    evaluation_instructions = []

    
    if args.datasetType == "humaneval":
                
        with open(args.evalFile, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                evaluation_instructions.append({
                    'task_id': item['task_id'],
                    'prompt': item['prompt'],
                    'instruction': item['prompt'],
                    'input': '',
                })
                
    elif args.datasetType == "IFEval":
        with open(args.evalFile, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                evaluation_instructions.append({
                    'key': item['key'],
                    'prompt': item['prompt'],
                    'instruction': item['prompt'],
                    'input':'',
                })
                

    elif args.datasetType == "humaneval_instruct":
        ds = load_dataset("codeparrot/instructhumaneval")
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        for item in ds[split]:
            evaluation_instructions.append({
                'task_id': item.get('task_id', item.get('name', f"task_{len(evaluation_instructions)}")),
                'prompt': item.get('instruction', item.get('instruction', '')),
                'instruction': item.get('instruction', item.get('prompt', '')),
                'input': '',
            })

    elif args.datasetType == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")
        split = 'test' if 'test' in ds else list(ds.keys())[0]
        for item in ds[split].select(range(100)):
            evaluation_instructions.append({
                'prompt': item.get('question', ''),
                'instruction': item.get('question', ''),
                'answer': item.get('answer', ''),
                'input': '',
            })

    else:
        with open(args.evalFile, 'r', encoding='utf-8') as f:
            data = json.load(f)
            for item in data:
                evaluation_instructions.append({
                    'instruction': item['instruction'],
                    'input': item.get('input', ''),
                })

    ## Loading reward model (only if BoN is enabled)
    reward_evaluator = None
    if use_bon:
        reward_model_args = Namespace(
            model_name_or_path="meta-llama/Llama-3.2-3b",
            trust_remote_code=False,
            cache_dir=None,
            model_max_length=512,
            full_finetune=False,
            adam8bit=False,
            double_quant=True,
            quant_type="nf4",
            bits=4,
            lora_modules=None,
            lora_r=64,
            lora_alpha=16,
            lora_dropout=0.0,
            gradient_checkpointing=True,
            do_train=False,
            fp16=True,
            bf16=False,
            checkpoint_dir=args.reward_model_checkpoint_dir,
        )

        reward_tokenizer = AutoTokenizer.from_pretrained(
            reward_model_args.model_name_or_path,
            cache_dir=None,
            model_max_length=512,
            padding_side="left",
            truncation_side="right",
        )

        config = RewardConfig(backbone_model_name_or_path=reward_model_args.model_name_or_path)

        if reward_tokenizer.pad_token is None:
            reward_tokenizer.pad_token_id = 0

        reward_model = RewardModel(
            args=reward_model_args,
            config=config,
            qlora=True,
            checkpoint_dir=reward_model_args.checkpoint_dir,
            adapter_name="lora_default",
        )

        reward_evaluator = RewardModelEvaluator(model=reward_model, tokenizer=reward_tokenizer)

    ### Loading generation model
    tokenizer = AutoTokenizer.from_pretrained(args.modelString, padding_side='left')
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token

    if not args.use_base_model:
        print(f"Loading model from checkpoint directory: {args.checkpoint_dir}");
        
        model = load_4bit_model_for_inference(
            checkpoint_dir=args.checkpoint_dir,
            bits=4,
            fp16=True,
            bf16=False,
            gradient_checkpointing=False,
            adapter_name="lora_policy",
            is_trainable=False,
            reuse_base_model=True,
            trust_remote_code=True,
            base_model_mapping=None,
            fully_initialize=False,
            base_model_name_or_path_for_fully_initialize=args.modelString,
        );
        
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.modelString, device_map='auto', quantization_config=bnb_config
        ).eval()
        print(f"Loaded model {args.modelString} with 4-bit quantization for inference.")

    with torch.no_grad():
        with torch.cuda.amp.autocast():

            all_outputs = []

            for gen_idx in range(args.numGenerations):
                print(f"\nGeneration round {gen_idx + 1}/{args.numGenerations}" +
                      (f" [BoN N={args.bon_n}]" if use_bon else " [standard]"))

                for i in tqdm.tqdm(range(0, len(evaluation_instructions), args.batchSize)):
                    batch = evaluation_instructions[i:i + args.batchSize]

                    if use_bon:
                        # --- BoN path ---
                        batch_outputs = best_of_n_sampling(
                            model=model,
                            tokenizer=tokenizer,
                            reward_evaluator=reward_evaluator,
                            items=batch,
                            dataset_type=args.datasetType,
                            n=args.bon_n,
                            max_retries=args.maxRetries,
                            response_len=args.response_len,
                            model_string=args.modelString,
                            gen_idx = gen_idx,
                            greedy = args.greedy,
                        )
                    else:
                        # --- Standard path (original behaviour) ---
                        prompts = build_prompts(batch, args.datasetType)

                        responses = generate_with_retry(
                            model, tokenizer, prompts,
                            max_retries=args.maxRetries,
                            max_new_tokens=args.response_len,
                        )

                        batch_outputs = [
                            build_output_record(
                                item=batch[j],
                                response=responses[j],
                                dataset_type=args.datasetType,
                                model_string=args.modelString,
                                gen_idx=gen_idx,
                                bon_score=None,  # No score in standard mode
                            )
                            for j in range(len(batch))
                        ]

                    all_outputs.extend(batch_outputs)

    ## Save results
    os.makedirs(os.path.dirname(args.outFile), exist_ok=True)

    if args.datasetType in ["humaneval", "humaneval_instruct", "gsm8k"]:
        with open(args.outFile, 'w', encoding='utf-8') as f:
            for output in all_outputs:
                f.write(json.dumps(output, ensure_ascii=False) + '\n')
    elif args.datasetType == "IFEval":
        with open(args.outFile, 'w', encoding='utf-8') as f:
            for row in all_outputs:
                f.write(json.dumps({
                    "prompt": row["prompt"],
                    "response": row["response"]
                }, ensure_ascii=False) + "\n")

    else:
        with open(args.outFile, 'w', encoding='utf-8') as f:
            json.dump(all_outputs, f, indent=2, ensure_ascii=False)

            
    print(f"\nBest-of-N sampling: {'ENABLED (N=' + str(args.bon_n) + ')' if use_bon else 'DISABLED'}")
    print(f"Generated completions for {len(evaluation_instructions)} prompts")
    print(f"Total outputs: {len(all_outputs)}")
    print(f"Saved to {args.outFile}")


if __name__ == "__main__":
    fire.Fire(main)