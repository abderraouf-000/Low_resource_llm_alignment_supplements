#!/bin/bash
set -euo pipefail

python Evaluations/TruthfulQAGeneration_MC1.py\
    --dataset_name "truthful_qa" \
    --indicesMod 1 \
    --indicesRemainder 0 \
    --batchSize 32 \
    --maxRetries 2 \
    --seed 0 \
    --modelString "meta-llama/Llama-3.2-3B" \
    "$@"