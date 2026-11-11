#!/bin/sh
# Next after H3 (Dr. Zia's order): tumour core (TC) and enhancing tumour (ET) on the same 30 test patients.
# Zero-shot with a box on every slice and with re-prompting every 30 slices, then the v2 adapter, which was
# trained on whole tumour only, to see whether it carries over. FLAIR input as before; T1c comes later (H2).
# Patients without ET are skipped for ET. About 4 hours on the M2. Safe to rerun if it stops.
set -e
COMMON="--data data --splits splits_base.json --subset test --limit 30 --norm slice --regions TC,ET --placement largest --direction both --resume"
python run.py eval $COMMON --mode per-slice --name tcet_ceiling_30
python run.py eval $COMMON --mode reprompt --step 30 --name tcet_rp30_30
python run.py eval $COMMON --mode reprompt --step 30 --lora lora_encoder_v2.pt --name tcet_lora_v2_rp30_30
python run.py summary results/tcet_ceiling_30.csv results/tcet_rp30_30.csv results/tcet_lora_v2_rp30_30.csv
