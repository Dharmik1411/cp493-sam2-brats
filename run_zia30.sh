#!/bin/sh
# Dr. Zia's three settings on the first 30 of his 150 seed-42 patients, whole tumour, Mac GPU
set -e
COMMON="--data data --splits splits_zia.json --subset test --limit 30 --norm slice --regions WT"
python run.py eval $COMMON --mode per-slice --name zia_ceiling_30
python run.py eval $COMMON --mode reprompt --placement largest --direction both --step 30 --name zia_rp30_30
python run.py eval $COMMON --mode reprompt --placement largest --direction both --step 10 --name zia_rp10_30
python run.py summary results/zia_ceiling_30.csv results/zia_rp30_30.csv results/zia_rp10_30.csv
