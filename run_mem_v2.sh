#!/bin/sh
# H4 re-check (Nov 11 meeting): the encoder adapter v2 got a longer training recipe than the memory adapter, so
# train the memory-attention adapter with the SAME recipe (450 steps, lr 3e-4, the 412 training patients) and
# compare the two on the 30 validation patients and on all 150 test patients; also add it to the H3 loose-box table.
# About 8.5 hours on the M2. Safe to rerun if it stops.
set -e
[ -f lora_memory_v2.pt ] || python train_lora.py --data data --splits splits_base.json --part memory_attention --steps 450 --lr 3e-4 --out lora_memory_v2.pt
RP30="--data data --splits splits_base.json --norm slice --regions WT --mode reprompt --placement largest --direction both --step 30 --resume"
python run.py eval $RP30 --subset val --lora lora_memory_v2.pt --name val_mem_v2
python run.py eval $RP30 --subset test --limit 30 --noise low --lora lora_memory_v2.pt --name h3_lora_mem_v2_rp30_low_30
python run.py eval $RP30 --subset test --limit 150 --lora lora_memory_v2.pt --name lora_memory_v2_rp30_150
python run.py summary results/val_base.csv results/val_enc_v2.csv results/val_mem_v2.csv
python analyze.py --all results/rp30_150.csv results/lora_memory_v2_rp30_150.csv results/lora_encoder_v2_rp30_150.csv results/ceiling_150.csv
