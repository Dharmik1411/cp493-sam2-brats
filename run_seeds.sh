#!/bin/sh
# H4 repeat with training seeds 1 and 2 (seed 0 is lora_memory.pt / lora_encoder.pt), as the proposal plans 3 seeds.
# Each adapter is tested exactly like rp30_30. About 8 hours on the M2.
set -e
LR=${LR:-1e-4}
COMMON="--data data --splits splits_base.json --subset test --limit 30 --norm slice --regions WT --mode reprompt --placement largest --direction both --step 30"
for s in 1 2; do
  for p in memory_attention:memory image_encoder:encoder; do
    part=${p%%:*}; name=${p##*:}
    [ -f lora_${name}_s$s.pt ] || python train_lora.py --data data --splits splits_base.json --part $part --steps 150 --lr $LR --seed $s --out lora_${name}_s$s.pt
    python run.py eval $COMMON --lora lora_${name}_s$s.pt --name lora_${name}_s${s}_rp30_30
  done
done
python run.py summary results/rp30_30.csv results/lora_*_rp30_30.csv
