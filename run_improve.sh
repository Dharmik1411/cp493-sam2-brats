#!/bin/sh
# Round 1 of trying to beat the current best: LoRA in the image encoder, 150 steps at lr 1e-4 on 112 patients.
# Choices are made on 30 validation patients (splits_zia.json "val": training-split patients that no adapter
# was trained on), so Dr. Zia's 150 test patients stay untouched until the final check.
# The change: 3x the steps, 3x the learning rate, and about 4x the training patients (412 instead of 112).
# Unpack new_members.txt first (see README). About 6.5 hours on the M2. Safe to rerun if it stops.
set -e
COMMON="--data data --splits splits_zia.json --subset val --norm slice --regions WT --mode reprompt --placement largest --direction both --step 30 --resume"
python run.py eval $COMMON --name val_base
python run.py eval $COMMON --lora lora_encoder.pt --name val_enc
[ -f lora_encoder_v2.pt ] || python train_lora.py --data data --splits splits_zia.json --part image_encoder --steps 450 --lr 3e-4 --out lora_encoder_v2.pt
python run.py eval $COMMON --lora lora_encoder_v2.pt --name val_enc_v2
python run.py summary results/val_base.csv results/val_enc.csv results/val_enc_v2.csv
