#!/bin/bash

set -euo pipefail

python Ground_truth_based_optimized_synthetic_generation.py \
    --response_len 128 \
    --resampling_number 5 \
    --promptsFile "data/sampled_alpaca_with_ids&prose.json" \
    --batchSize 46 \
    --maxRetries 3 \
    --seed 0 \
    --modelString "meta-llama/Llama-3.2-3B" \
    --maxResamplingAttempts 3 \
    "$@"