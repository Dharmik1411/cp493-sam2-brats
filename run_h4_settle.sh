#!/bin/sh
# Settling H4 (Nov 17). The v2 recipe (450 steps, lr 3e-4) was picked on validation for the encoder and it hurt
# the memory adapter, so here the memory adapter gets the same chance the encoder had:
#   1. memory v1 (150 steps, lr 1e-4, 112 patients) on the validation patients and on all 150 test patients,
#      for a same-recipe comparison with encoder v1 on all 150 (lora_encoder_rp30_150)
#   2. memory long_lowlr (450 steps on the 412 patients, at the lower lr 1e-4), on validation and all 150
# The memory recipe with the best validation Dice is then memory's best, compared with encoder v2 (the encoder's
# best on validation) on the 150 test patients. About 12 hours on the M2. Safe to rerun if it stops.
set -e
RP30="--data data --splits splits_base.json --norm slice --regions WT --mode reprompt --placement largest --direction both --step 30 --resume"
python run.py eval $RP30 --subset val --lora lora_memory.pt --name val_mem
[ -f lora_memory_long_lowlr.pt ] || python train_lora.py --data data --splits splits_base.json --part memory_attention --steps 450 --lr 1e-4 --out lora_memory_long_lowlr.pt
python run.py eval $RP30 --subset val --lora lora_memory_long_lowlr.pt --name val_mem_long_lowlr
python run.py summary results/val_base.csv results/val_mem.csv results/val_mem_v2.csv results/val_mem_long_lowlr.csv results/val_enc.csv results/val_enc_v2.csv
[ -f results/lora_memory_rp30_150.csv ] || cp results/lora_memory_rp30_30.csv results/lora_memory_rp30_150.csv  # reuse the first 30
python run.py eval $RP30 --subset test --limit 150 --lora lora_memory.pt --name lora_memory_rp30_150
python run.py eval $RP30 --subset test --limit 150 --lora lora_memory_long_lowlr.pt --name lora_memory_long_lowlr_rp30_150
python analyze.py --all results/rp30_150.csv results/lora_encoder_rp30_150.csv results/lora_memory_rp30_150.csv \
  results/lora_encoder_v2_rp30_150.csv results/lora_memory_v2_rp30_150.csv results/lora_memory_long_lowlr_rp30_150.csv
