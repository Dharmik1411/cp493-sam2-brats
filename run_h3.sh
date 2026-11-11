#!/bin/sh
# H3: loose boxes at Ludwig et al.'s two noise levels (low = 30% bigger, up to 20% shift; medium = 10%, 10%)
# on the same 30 patients, WT. Clean-box results already exist (ceiling_30, rp30_30).
#   per-slice vs Dr. Zia's rp30: does a loose box cost more Dice with propagation?
#   one box propagated through the whole volume: where do over-propagation errors land (fp_near column)?
#   the best LoRA adapter (v2) with the loosest boxes: does its gain survive?
# Loosest level first, so a partial run is still useful. About 7 to 8 hours on the M2.
set -e
# --resume: if it stops, run the same command again and it carries on
COMMON="--data data --splits splits_base.json --subset test --limit 30 --norm slice --regions WT --placement largest --direction both --resume"
for n in low medium; do
  python run.py eval $COMMON --mode per-slice --noise $n --name h3_ceiling_${n}_30
  python run.py eval $COMMON --mode reprompt --step 30 --noise $n --name h3_rp30_${n}_30
done
for n in none low; do
  python run.py eval $COMMON --mode video --noise $n --name h3_onebox_${n}_30
done
# the best adapter (v2, image encoder) with the loosest boxes; its clean-box result on these 30 is in lora_encoder_v2_rp30_150.csv
python run.py eval $COMMON --mode reprompt --step 30 --noise low --lora lora_encoder_v2.pt --name h3_lora_v2_rp30_low_30
python run.py summary results/ceiling_30.csv results/rp30_30.csv results/h3_*.csv
