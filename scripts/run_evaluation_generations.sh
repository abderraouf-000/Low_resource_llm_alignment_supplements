#!/bin/bash
set -euo pipefail

python Evaluations/Checkpoint_generations_for_evaluation.py \
    --instruction_field "instruction" \
    --Human_eval_file "Evaluations/Eval_data/human-eval-v2-20210705.jsonl" \
    --Alpaca_eval_file "Evaluations/Eval_data/alpaca_eval.json" \
    --outFile "Evaluations/Generations/Human_eval_completion_generation_base_greedy_len256.jsonl" \
    --modelString "meta-llama/Llama-3.2-3B" \
    --dataset_type "alpaca" \
    --num_generations 1 \
    --max_retries 8 \
    --response_len 256 \
    "$@"