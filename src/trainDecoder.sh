#!/bin/bash

python train/trainDecoder.py \
    --epochs 5 \
    --batch_size 4 \
    --embeddings E:/datasets/mimic/embeddings/cls4-mixer-lora_train.pkl \
    --normalize \
    --append_eos \
    --dataset petro \
    --save_path checkpoints/test \
    --model_name meta-llama/Llama-3.2-1B



