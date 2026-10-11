#!/bin/bash
set -euo pipefail
python Synthetic_post_process.py \
  --data_to_merge "" \
  --do_balance "true" \
  --max_per_class 30000 \
  --do_synthetic "False" \
  --seed 42 \
  "$@"