#!/bin/sh
# H4 first results: LoRA in the memory attention vs the image encoder (matched size, ~117K params each),
# then both tested exactly like rp30_30 (bidirectional, re-prompt every 30, whole tumour, 30 patients).
# STEPS and LR are first guesses; time 5 steps first:  python train_lora.py ... --steps 5
set -e
# an adapter whose .pt already exists is not retrained; delete the .pt to retrain it
STEPS=${STEPS:-150}
LR=${LR:-1e-4}  # the value used for every v1 adapter in results/
[ -f lora_memory.pt ] || python train_lora.py --data data --splits splits_base.json --part memory_attention --steps $STEPS --lr $LR --out lora_memory.pt
[ -f lora_encoder.pt ] || python train_lora.py --data data --splits splits_base.json --part image_encoder --steps $STEPS --lr $LR --out lora_encoder.pt
COMMON="--data data --splits splits_base.json --subset test --limit 30 --norm slice --regions WT --mode reprompt --placement largest --direction both --step 30"
python run.py eval $COMMON --lora lora_memory.pt --name lora_memory_rp30_30
python run.py eval $COMMON --lora lora_encoder.pt --name lora_encoder_rp30_30
python run.py summary results/rp30_30.csv results/lora_memory_rp30_30.csv results/lora_encoder_rp30_30.csv
