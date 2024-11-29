#!/bin/bash
cd /mloscratch/homes/shcherba
source ~/.zshrc
source set_env.sh
cd dl-char-llm
conda activate char-llm
PYTHONPATH='/mloscratch/homes/shcherba/dl-char-llm' torchrun --nproc_per_node=1 train.py \
    "$@" \
    > /mloscratch/homes/shcherba/output_$1.log 2>&1
