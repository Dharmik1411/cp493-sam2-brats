#!/bin/sh
# The winning H4 setting (LoRA in the image encoder, seed 0) on all 150 of Dr. Zia's seed-42 patients,
# next to his rp30 baseline and the box-on-every-slice ceiling, whole tumour.
# Each *_150.csv starts as a copy of the 30-patient file and --resume skips those 30, so only the other
# 120 patients run per setting. If it stops, run the same command again: it carries on where it was.
# About 10 hours on the M2.
set -e
for s in zia_rp30 lora_encoder_rp30 zia_ceiling; do
  [ -f results/${s}_150.csv ] || cp results/${s}_30.csv results/${s}_150.csv
done
COMMON="--data data --splits splits_zia.json --subset test --limit 150 --norm slice --regions WT --resume"
RP30="--mode reprompt --placement largest --direction both --step 30"
python run.py eval $COMMON $RP30 --name zia_rp30_150
python run.py eval $COMMON $RP30 --lora lora_encoder.pt --name lora_encoder_rp30_150
python run.py eval $COMMON --mode per-slice --name zia_ceiling_150
python run.py summary results/zia_rp30_150.csv results/lora_encoder_rp30_150.csv results/zia_ceiling_150.csv
