#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python train.py --dataset custom \
                --epochs 50 \
                --batch_size 512 \
                --learning_rate 1e-4 \
                --temperature 5e-2 \
                --eval_every 1 \
                --patience 40 \
                --optimizer RAdam \
                --anneal_every 10 \
                --anneal_factor 0.5 \
                --init_size 0.05 \
                --num_samples 2000000 \
                --rank 2 \
                --seed 0