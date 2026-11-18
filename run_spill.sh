#!/bin/sh
# H3, where over-propagation lands (Dr. Zia, Nov 14): one box on the largest-tumour slice carried through the whole
# scan, with clean and loose boxes, first 30 test patients. The spill_* columns give the share of voxels predicted
# 1-3, 4-10, 11-20 and 21+ slices past the tumour's ends, and spill_max the furthest one. About 4 hours on the M2.
set -e
COMMON="--data data --splits splits_base.json --subset test --limit 30 --norm slice --regions WT --placement largest --direction both --resume"
for n in none low; do
  python run.py eval $COMMON --mode video --noise $n --name h3_onebox_spill_${n}_30
done
python analyze_h3.py
