#!/bin/bash
cd /mloscratch/homes/shcherba
source ~/.zshrc
source set_env.sh
cd dl-char-llm
conda activate char-llm
sanitized_arg=$(echo "$1" | tr '/' '_')
PYTHONPATH='/mloscratch/homes/shcherba/dl-char-llm' torchrun --nproc_per_node=1 train.py \
    "$@" \
    > /mloscratch/homes/shcherba/output_${sanitized_arg}.log 2>&1
