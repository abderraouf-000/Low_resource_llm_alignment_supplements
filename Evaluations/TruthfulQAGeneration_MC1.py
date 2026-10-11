from argparse import Namespace

from datasets import load_dataset
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import transformers
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
from models.qlora_model import load_4bit_model_for_inference
import torch.nn.functional as F
import numpy as np

import argparse
import json
import random
from collections import defaultdict


HF_TOKEN = os.getenv("HF_TOKEN")
login(token=HF_TOKEN)


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

def _parse_indices_helper(indices):
    for index in indices.split(','):
        if '-' in index:
            start, end = index.split('-')
            for i in range(int(start), int(end) + 1):
                yield i
        else:
            yield int(index)

def parse_indices(indices):
    return list(_parse_indices_helper(indices))



def get_answer_logprob(model, tokenizer, question, answer):
    """Calculate log-probability of answer given question"""
    # Format the input - simple Q: A: format

    prompt = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.\n\n### Instruction:\n{question}\n\n### Response: {answer}";
    
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to('cuda') for k, v in inputs.items()}
    
    # Get model outputs
    with torch.no_grad():
        outputs = model(**inputs,)
        logits = outputs.logits
    

    
    question_part = f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request, you should provide a response to the user's query.\n\n### Instruction:\n{question}\n\n### Response: ";

    
    
    question_tokens = tokenizer(question_part, return_tensors="pt")
    question_length = question_tokens['input_ids'].shape[1]
    
    # Get answer token positions (excluding the prompt)
    answer_start = question_length - 1  # -1 because we want to predict from the last question token
    answer_tokens = inputs['input_ids'][0][answer_start:]
    
    
    # Get log-probabilities for answer tokens
    log_probs = F.log_softmax(logits[0], dim=-1)
    
    # Calculate average log-probability for the answer
    answer_log_probs = []
    for i, token_id in enumerate(answer_tokens[1:]):  # Skip first token (it's the one we're predicting from)
        if i + answer_start < log_probs.shape[0]:
            token_log_prob = log_probs[i + answer_start, token_id].item()
            answer_log_probs.append(token_log_prob)
    
    # Return average log-probability
    return np.mean(answer_log_probs) if answer_log_probs else float('-inf');

    

def evaluate_mc1_question(model, tokenizer, question, choices, correct_answer_index):
    """Evaluate a single MC1 question using log-probability scoring"""
    log_probs = []
    
    for choice in choices:
        log_prob = get_answer_logprob(model, tokenizer, question, choice)
        log_probs.append(log_prob)
    
    # Find the choice with maximum log-probability
    predicted_index = np.argmax(log_probs)
    is_correct = predicted_index == correct_answer_index
    
    return {
        'predicted_index': predicted_index,
        'correct_index': correct_answer_index,
        'is_correct': is_correct,
        'log_probs': log_probs,
        'choices': choices
    };


    

def load_truthfulqa_mc1_dataset():
    """Load TruthfulQA MC1 dataset"""

    dataset = load_dataset("norabelrose/truthful_qa")['validation']
    
    questions = []
    for item in dataset:
        questions.append({
            'question': item['question'],
            'choices': item['choices'],
            'correct_answer_index': 0 # Find the correct answer
        })
    
    print(f"Loaded {len(questions)} MC1 questions")
    return questions



    

def main(
    use_base_model = True,
    checkpoint_dir = "",
    dataset_name: str = "truthful_qa",
    indices: str = None,
    indicesMod: int = 1,
    indicesRemainder: int = 0,
    batchSize: int = 32,  # Reduced batch size for MC evaluation
    maxRetries: int = 2,
    outDir: str = "Evaluations/Generations/truthfulQA-mc1-logprob-",
    seed: int = 0,
    modelString: str = "meta-llama/Llama-3.2-3B"
):
    

    args = Namespace(
        use_base_model = use_base_model,
        checkpoint_dir = checkpoint_dir,
        dataset_name=dataset_name,
        indices=indices,
        indicesMod=indicesMod,
        indicesRemainder=indicesRemainder,
        batchSize=batchSize,
        maxRetries=maxRetries,
        outDir=outDir,
        seed=seed,
        modelString=modelString
    );
    
    
    # Create output directory
    os.makedirs(args.outDir, exist_ok=True)
    
    # Load TruthfulQA MC1 dataset
    questions = load_truthfulqa_mc1_dataset()
    
    
    # Initialize tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(args.modelString, padding_side='left')
    tokenizer.padding_side = "left" 
    tokenizer.pad_token = tokenizer.eos_token

    if args.use_base_model:
        model = AutoModelForCausalLM.from_pretrained(args.modelString, device_map='auto', quantization_config=bnb_config).eval()
    else:
        model = load_4bit_model_for_inference(
        checkpoint_dir = args.checkpoint_dir,
        # checkpoint_dir = f"PPO/Llama3.2-3b-ppo-rulebased-level3_checkpoint350_prose&nonprose/ppo-checkpoint-250/adapter_model/lora_policy",
        # checkpoint_dir = f"Instruction_following_alpaca/checkpoint-9753/lora_policy",
        bits=  4,
        fp16 = False,
        bf16 = True,
        gradient_checkpointing = False,
        adapter_name="lora_policy",
        is_trainable=False,
        reuse_base_model=True,
        trust_remote_code=True,
        base_model_mapping=None,
        fully_initialize=False,
        base_model_name_or_path_for_fully_initialize = "meta-llama/Llama-3.2-3B",
    );


    
    indices = parse_indices(args.indices) if args.indices is not None else list(range(len(questions)))
    
    results = []
    correct_predictions = 0
    total_predictions = 0
    
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            
            for idx in tqdm.tqdm(indices):
                if idx >= len(questions):
                    continue
                if idx % args.indicesMod != args.indicesRemainder:
                    continue
                if os.path.exists(os.path.join(args.outDir, f'{idx}.json')):
                    # Load existing result
                    with open(os.path.join(args.outDir, f'{idx}.json'), 'r') as f:
                        existing_result = json.load(f)
                        if existing_result.get('is_correct') is not None:
                            correct_predictions += existing_result['is_correct']
                            total_predictions += 1
                    continue
                
                question_data = questions[idx]
                
                # Evaluate the question
                result = evaluate_mc1_question(
                    model, tokenizer, 
                    question_data['question'], 
                    question_data['choices'], 
                    question_data['correct_answer_index']
                );

                
                
                # Track accuracy
                correct_predictions += result['is_correct']
                total_predictions += 1
                
                # Save result
                save_dict = {
                    'question': question_data['question'],
                    'choices': question_data['choices'],
                    'correct_answer_index': int(question_data['correct_answer_index']),
                    'predicted_index': int(result['predicted_index']),
                    'predicted_answer': question_data['choices'][int(result['predicted_index'])],
                    'correct_answer': question_data['choices'][int(question_data['correct_answer_index'])],
                    'is_correct': bool(result['is_correct']),
                    'log_probs': [float(lp) for lp in result['log_probs']],
                    'accuracy_so_far': float(correct_predictions / total_predictions)
                }
                
                with open(os.path.join(args.outDir, f'{idx}.json'), 'w') as f:
                    json.dump(save_dict, f, indent=2)
                
                results.append(save_dict)
                
                # Print progress
                if total_predictions % 200 == 0:
                    print(f"Progress: {total_predictions} questions, Accuracy: {correct_predictions/total_predictions:.3f}")
    
    # Final accuracy calculation
    final_accuracy = float(correct_predictions / total_predictions) if total_predictions > 0 else 0
    print(f"Final TruthfulQA MC1 Accuracy: {final_accuracy:.3f} ({correct_predictions}/{total_predictions})")
    
    # Save summary
    os.makedirs(args.outDir, exist_ok=True)
    
    summary = {
        'total_questions': int(total_predictions),
        'correct_predictions': int(correct_predictions),
        'accuracy': float(final_accuracy),
        'model': args.modelString,
        'dataset': args.dataset_name
    }
    
    with open(os.path.join(args.outDir, 'summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    fire.Fire(main)