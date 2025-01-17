#!/bin/bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
python train.py --dataset custom \
                --epochs 10 \
                --batch_size 512 \
                --learning_rate 5e-4 \
                --temperature 1e-2 \
                --eval_every 1 \
                --patience 40 \
                --optimizer RAdam \
                --anneal_every 10 \
                --anneal_factor 0.5 \
                --init_size 0.05 \
                --num_samples 10000000\
                --rank 3 \
                --seed 0
