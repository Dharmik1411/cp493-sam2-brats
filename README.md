# Prompting or Training? SAM 2 and LoRA on BraTS 2021 brain tumours

CP493 Directed Studies, Wilfrid Laurier University, Fall 2026. Dharmik Patel, supervised by Dr. Zia Ud Din.

SAM 2 can segment a 3D MRI scan by treating its slices as video frames: a box on a few slices, and its memory
carries the mask through the rest. Accuracy drops with distance from the boxed slice. This repo

1. reproduces Dr. Zia's zero-shot protocol on BraTS 2021 (150 patients, seed 42, whole tumour, FLAIR):
   propagation in both directions from the largest-tumour slice, a fresh box and memory reset every 30 slices;
2. tests whether a small LoRA adapter recovers the lost accuracy, and whether it works better in SAM 2's
   memory attention or in its image encoder (same number of trainable parameters);
3. (next) tests how loose boxes affect propagation (`run_h3.sh`).

## Main result: 150 test patients, whole tumour, mean ± SD

| Setting | Boxes per patient | Dice | HD95 (mm) | NSD (1 mm) |
|---|---|---|---|---|
| Zero-shot, re-prompt every 30 slices | 2.3 | 0.808 ± 0.112 | 11.4 ± 8.3 | 0.479 ± 0.154 |
| LoRA in image encoder, v1 (150 steps, 112 training patients) | 2.3 | 0.847 ± 0.103 | 10.3 ± 8.0 | 0.610 ± 0.150 |
| LoRA in image encoder, v2 (450 steps, 412 training patients) | 2.3 | **0.887 ± 0.087** | 9.2 ± 9.6 | **0.697 ± 0.142** |
| Zero-shot, box on every tumour slice | 67.3 | 0.865 ± 0.089 | 7.4 ± 5.8 | 0.621 ± 0.146 |

Dr. Zia's paper (SAM 2.1 Hiera-Large) reports 0.813 Dice at 2.4 boxes and 0.862 / 7.3 mm / 0.621 for a box on
every slice. This repo uses Hiera-Small. Per-patient results for every run are in `results/`.

## Files

| File | What it does |
|---|---|
| `brats.py` | loads a case, makes box prompts (clean or loose), plans the re-prompt segments, Dice / HD95 / NSD / over-propagation |
| `sam2_runner.py` | SAM 2.1 with a box on every slice, with propagation, or with Dr. Zia's re-prompting; loads LoRA weights |
| `lora.py` | LoRA layers for the memory attention or the image encoder, rank matching, self-check (`python lora.py`) |
| `train_lora.py` | trains one adapter on 8-slice clips from training patients |
| `run.py` | `smoke` (check the install), `split` (make the patient split), `eval` (one setting to a CSV), `summary` |
| `analyze.py` | paired comparisons between results files: Wilcoxon signed-rank, Holm correction, bootstrap 95% CI |
| `make_figures.py`, `figures/` | the report's figures, made from `results/` |
| `run_*.sh` | one script per experiment, listed below in the order they were run |
| `splits_base.json` | patient IDs: `test` (Dr. Zia's 150), `val` (30), `train` |
| `members.txt`, `new_members.txt` | listing of the dataset archive, and the 330 extra patients unpacked for v2 |
| `lora_*.pt`, `lora_*_log.csv` | trained adapters (about 0.5 MB each, each stores the patients it was trained on) and their loss per step |
| `results/*.csv` | one row per patient: Dice, HD95, NSD, boxes used, over-propagation |

## Setup (Python 3.10+)

```bash
python3 -m venv .venv && source .venv/bin/activate
SAM2_BUILD_CUDA=0 pip install -r requirements.txt
python run.py smoke     # downloads SAM 2.1-Small, checks the metric functions, segments a synthetic tumour
python lora.py          # checks the adapters: matched sizes, output unchanged before training, gradients only in LoRA
```

Uses an NVIDIA GPU, an Apple Silicon GPU (MPS) or the CPU, whichever is there. All results here were made on an
Apple M2. On Apple GPUs SAM 2 prints that it is skipping a post-processing step: its small hole-filling step needs
a CUDA extension. It is skipped the same way in every run, so comparisons between runs are not affected.

## Data

BraTS 2021 training data (1,251 adult glioma patients with labels), Kaggle copy
[`dschettler8845/brats-2021-task1`](https://www.kaggle.com/datasets/dschettler8845/brats-2021-task1) (13.4 GB).
The scans are not in this repo.

```bash
# download archive.zip from the Kaggle page, then either unpack everything (about 10 GB)
mkdir -p data
unzip -p ~/Downloads/archive.zip BraTS2021_Training_Data.tar | tar -x -C data

# or unpack only the patients used (the split is already in splits_base.json; to check it,
# python run.py split --data members.txt --sample 150 --seed 42 --splits check.json  gives the same 150 test IDs)
python -c "import json;[print('./'+n+'/') for n in json.load(open('splits_base.json'))['test']]" > test.txt
grep -F -f test.txt members.txt > test_members.txt
unzip -p ~/Downloads/archive.zip BraTS2021_Training_Data.tar | tar -x -C data -T test_members.txt
unzip -p ~/Downloads/archive.zip BraTS2021_Training_Data.tar | tar -x -C data -T new_members.txt   # val + extra training
```

`splits_base.json` in this repo already holds the split used for every result. Notes on it:

- `test`: 150 patients drawn with `random.Random(42).sample` from the sorted patient IDs. Never used for training.
- `val`: 30 training-split patients that no adapter was trained on, used to choose between adapter versions
  (drawn with `random.Random(2026)`, together with the 300 extra training patients in `new_members.txt`).
  `train_lora.py` never trains on `test` or `val` patients.
- Training uses every `train` patient unpacked in `data/`: 112 for the v1 adapters, 412 for v2.

## Reproducing the results

Each script prints a summary when it ends and is safe to rerun if it stops (`--resume` skips patients already in a
results file). Times are on an Apple M2.

| Step | Command | What it gives | Time |
|---|---|---|---|
| 1 | `sh run_base_30.sh` | Dr. Zia's 3 zero-shot settings on the first 30 test patients | about 3 h |
| 2 | `LR=1e-4 sh run_lora.sh` | H4: LoRA in memory attention vs image encoder, seed 0, tested on the same 30 | about 3.5 h |
| 3 | `sh run_seeds.sh` | H4 repeated with training seeds 1 and 2 | about 8 h |
| 4 | `sh run_150.sh` | zero-shot re-prompting, LoRA v1 and box on every slice on all 150 test patients | about 10 h |
| 5 | `sh run_improve.sh` | v2 adapter (450 steps, lr 3e-4, 412 patients), compared on the 30 val patients | about 6.5 h |
| 6 | `sh run_v2_150.sh` | v2 on all 150 test patients | about 4.5 h |
| 7 | `sh run_h3.sh` | H3: loose boxes at two levels, per-slice vs re-prompting, one box through the whole volume | about 8 h |

On a Mac, put `caffeinate -i` in front so it does not sleep. To look at any results file:
`python run.py summary results/<name>.csv`.

The statistics in the report come from `analyze.py`, for example the main table:

```bash
python analyze.py --all results/rp30_150.csv results/lora_encoder_rp30_150.csv \
    results/lora_encoder_v2_rp30_150.csv results/ceiling_150.csv
```

The figures: `python make_figures.py`.

Main options of `run.py eval`: `--mode per-slice | video | reprompt`, `--step` (re-prompt interval),
`--placement largest | middle | first-every`, `--norm slice` (FLAIR rescaled per slice from its 1st to 99th
percentile, as in Dr. Zia's paper), `--noise none | medium | low` (loose boxes), `--lora <weights.pt>`,
`--subset test | val`, `--limit N`, `--resume`.

Boxes are drawn from the ground-truth mask (no real users), so all numbers are upper bounds for real use.

## Use of AI

I used an AI assistant (Claude) to help write, debug and run this code and its experiments, as allowed by the
course policy.
