#!/bin/sh
# Final check of the round 1 winner (lora_encoder_v2.pt, picked on the 30 validation patients) on all 150 of
# Dr. Zia's test patients, next to the results already there. About 4.5 hours on the M2. Safe to rerun if it stops.
set -e
python run.py eval --data data --splits splits_zia.json --subset test --limit 150 --norm slice --regions WT \
  --mode reprompt --placement largest --direction both --step 30 --resume --lora lora_encoder_v2.pt --name lora_encoder_v2_rp30_150
python run.py summary results/zia_rp30_150.csv results/lora_encoder_rp30_150.csv results/lora_encoder_v2_rp30_150.csv results/zia_ceiling_150.csv
