#!/bin/sh
# H3: loose boxes at Ludwig et al.'s two noise levels (low = 30% bigger, up to 20% shift; medium = 10%, 10%)
# on the same 30 patients, WT. Clean-box results already exist (zia_ceiling_30, zia_rp30_30).
#   per-slice vs Dr. Zia's rp30: does a loose box cost more Dice with propagation?
#   one box propagated through the whole volume: where do over-propagation errors land (fp_near column)?
#   both LoRA adapters with the loosest boxes: does the H4 gain survive?
# Loosest level first, so a partial run is still useful. About 7 to 8 hours on the M2.
set -e
COMMON="--data data --splits splits_zia.json --subset test --limit 30 --norm slice --regions WT --placement largest --direction both"
for n in low medium; do
  python run.py eval $COMMON --mode per-slice --noise $n --name h3_ceiling_${n}_30
  python run.py eval $COMMON --mode reprompt --step 30 --noise $n --name h3_rp30_${n}_30
done
for n in none low; do
  python run.py eval $COMMON --mode video --noise $n --name h3_onebox_${n}_30
done
for a in memory encoder; do
  python run.py eval $COMMON --mode reprompt --step 30 --noise low --lora lora_${a}.pt --name h3_lora_${a}_rp30_low_30
done
python run.py summary results/zia_ceiling_30.csv results/zia_rp30_30.csv results/h3_*.csv
