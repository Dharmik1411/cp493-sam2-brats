#!/bin/sh
# H3 on all 150 test patients (Dr. Zia, Nov 14). Each *_150.csv starts as a copy of its 30-patient file and
# --resume skips those 30, so only the other 120 patients run. Loose boxes first (the main test), then both
# adapters with loose boxes, then medium boxes. About 16 hours on the M2, so two nights: if it stops, run the
# same command again and it carries on. Ends by printing the H3 tables and tests (analyze_h3.py).
set -e
for s in h3_ceiling_low h3_rp30_low h3_lora_v2_rp30_low h3_lora_mem_v2_rp30_low h3_ceiling_medium h3_rp30_medium; do
  if [ -f results/${s}_30.csv ] && [ ! -f results/${s}_150.csv ]; then cp results/${s}_30.csv results/${s}_150.csv; fi
done
COMMON="--data data --splits splits_base.json --subset test --limit 150 --norm slice --regions WT --placement largest --direction both --resume"
python run.py eval $COMMON --mode per-slice --noise low --name h3_ceiling_low_150
python run.py eval $COMMON --mode reprompt --step 30 --noise low --name h3_rp30_low_150
python run.py eval $COMMON --mode reprompt --step 30 --noise low --lora lora_encoder_v2.pt --name h3_lora_v2_rp30_low_150
if [ -f lora_memory_v2.pt ]; then
  python run.py eval $COMMON --mode reprompt --step 30 --noise low --lora lora_memory_v2.pt --name h3_lora_mem_v2_rp30_low_150
fi
python run.py eval $COMMON --mode per-slice --noise medium --name h3_ceiling_medium_150
python run.py eval $COMMON --mode reprompt --step 30 --noise medium --name h3_rp30_medium_150
python analyze_h3.py
