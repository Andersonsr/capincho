#!/bin/bash

python train/trainDecoder.py \
    --epochs 5 \
    --batch_size 4 \
    --embeddings E:/embeddings/coco/dinov3_train.pkl \
    --normalize \
    --append_eos \
    --dataset coco \
    --save_path checkpoints/dinov3-opt350m \
    --model_name facebook/opt-350m \
    --fsdp
#    --model_name meta-llama/Llama-3.2-1B



