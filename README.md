# hurricane-ri-forecast

Research project on **probabilistic 24-hour hurricane rapid intensification (RI)** using NOAA storm-centered satellite infrared imagery and environmental fields from **TC PRIMED**.

**RI definition used here:** maximum sustained wind increases by **≥ 30 kt** in **24 hours**.

<p align="center">
  <img src="assets/hero_project_overview.jpg" alt="Storm-centered infrared thumbnail for hurricane rapid intensification forecasting" width="920"/>
</p>

## Research question

> Does raw storm-centered satellite imagery add RI forecast skill beyond environmental and infrared-summary predictors (SHIPS-like tabular inputs)?

This is a research ablation study. The goal is **not** to beat operational NOAA products from public data alone. The goal is a fair, reproducible answer about whether imagery helps, and when it fails.

## What this repo contains

- End-to-end pipeline: download → prepare → train → evaluate → compare
- NOAA **TC PRIMED** ingestion (NetCDF overpasses + environmental diagnostics)
- Three-model ablation:
  - `tabular`: environment + IR summary features
  - `imagery`: CNN on storm-centered `IRWIN`
  - `fusion`: gated tabular + imagery model
- Season-held-out evaluation (train/val on 2017–2022, test on 2023–2024)
- Probabilistic metrics: Brier score, Brier skill vs climatology, bootstrap intervals, reliability
- Calibration audit: temperature scaling (primary) + isotonic (probability-quality check)
- Basin breakdown (AL vs EP)
- Case analysis: hit, miss, and false alarm
- Resume-friendly prepare (storm checkpoints + optional `--shard 0/2` parallel workers)

## Dataset

Source: [TC PRIMED](https://rammb-data.cira.colostate.edu/tcprimed/) (NOAA / CIRA), public AWS bucket:

`s3://noaa-nesdis-tcprimed-pds`

Default scope (`configs/default.yaml`):

| Split | Seasons | Basins |
|---|---|---|
| Train / val | 2017–2022 | AL, EP |
| Test | 2023–2024 | AL, EP |

Prepared experiment size (current run):

- **33,878** samples
- **1,409** RI events (~4.2%)
- splits: train 18,986 / val 6,779 / test 8,113

Raw layout:

```text
data/raw/tcprimed/<year>/<basin>/<storm_id>/
```

## Main results (held-out 2023–2024)

### Raw (uncalibrated)

| Model | Brier ↓ | BSS vs climatology | POD | Success ratio |
|---|---:|---:|---:|---:|
| tabular | 0.074 | -0.603 | 0.702 | 0.270 |
| imagery | **0.049** | -0.068 | 0.031 | 0.231 |
| fusion | 0.056 | -0.211 | 0.356 | 0.303 |

### Temperature-scaled (primary / operational)

| Model | T | Brier ↓ | BSS | POD | Success ratio |
|---|---:|---:|---:|---:|---:|
| tabular | 1.27 | 0.073 | -0.583 | 0.702 | 0.270 |
| imagery | 3.56 | **0.046** | -0.007 | 0.031 | 0.231 |
| fusion | 1.94 | 0.053 | -0.146 | 0.356 | 0.303 |

### Isotonic audit (probability quality; not used for detection claims)

| Model | Brier ↓ | BSS | POD |
|---|---:|---:|---:|
| tabular | **0.037** | **+0.194** | 0.000 |
| imagery | 0.045 | +0.035 | 0.000 |
| fusion | 0.040 | +0.136 | 0.010 |

Reading this honestly:

- **Primary fusion beats tabular** on BSS (`ΔBSS ≈ +0.44`), but **no primary model beats climatology** yet.
- Best primary Brier is imagery, mostly by almost never predicting RI (POD 0.03).
- **Temperature scaling** softens overconfidence (especially imagery, `T≈3.6`) while **preserving POD**.
- **Isotonic wins Brier** by shrinking probabilities below 0.5 → POD collapses. Better-calibrated risk, not better RI detection.
- Failure modes stay opposite: tabular over-fires; imagery under-fires; fusion sits in between.
- Accuracy is **not** the headline metric: RI is rare (~5%).

### Basin breakdown (temperature / primary)

| Model | Basin | n | RI rate | Brier | BSS | POD | Success |
|---|---|---:|---:|---:|---:|---:|---:|
| tabular | AL | 4999 | 3.1% | 0.064 | **-1.11** | 0.62 | 0.20 |
| tabular | EP | 3114 | 7.6% | 0.088 | -0.24 | 0.76 | 0.33 |
| imagery | AL | 4999 | 3.1% | 0.034 | -0.11 | 0.00 | 0.00 |
| imagery | EP | 3114 | 7.6% | 0.067 | **+0.05** | 0.05 | 0.43 |
| fusion | AL | 4999 | 3.1% | 0.045 | -0.50 | 0.33 | 0.20 |
| fusion | EP | 3114 | 7.6% | 0.065 | **+0.08** | 0.37 | 0.42 |

EP has ~2.5× the AL RI base rate. Fusion’s skill is **EP-driven**; on AL, tabular over-fires hardest and fusion only partially corrects it.

### Where imagery helps / fails (temperature / primary)

**By intensity** (test RI rate in parentheses):

| Regime | Tabular BSS | Imagery BSS | Fusion BSS | Takeaway |
|---|---:|---:|---:|---|
| Cat1–2 64–95 kt (10.7% RI) | -0.66 | **+0.06** | ≈0.00 | **Imagery’s only clear win**: tabular over-fires; fusion mostly fixes it |
| TS 50–63 kt (6.9%) | -0.55 | ≈0.00 | -0.22 | Imagery near climatology; fusion still better than tabular |
| TS &lt;50 kt (3.0%) | -0.43 | -0.04 | -0.27 | Same pattern, lower base rate |
| Cat3+ ≥96 kt (1.8%) | -2.54 | -0.98 | -0.54 | Rare RI; tabular still fires (POD 0.31, succ 0.07) |

**By shear:**

| Regime | n | RI rate | Note |
|---|---:|---:|---|
| low &lt;10 kt | 5552 | 6.8% | Almost all RI lives here; tabular over-fires (BSS -0.62); fusion improves but still &lt; clim |
| mod 10–20 kt | 2130 | 0.6% | Very rare RI; fusion/imagery silent; tabular almost silent |
| high ≥20 kt | 431 | **0%** | No RI in held-out test; models stay quiet (physically sensible) |

### Reliability (held-out test)

<p align="center">
  <img src="assets/reliability_tabular_default.png" alt="Tabular reliability" width="32%"/>
  <img src="assets/reliability_imagery_default.png" alt="Imagery reliability" width="32%"/>
  <img src="assets/reliability_fusion_default.png" alt="Fusion reliability" width="32%"/>
</p>

<p align="center"><em>Left to right: tabular, imagery, fusion (temperature-scaled primary). Overconfidence is reduced vs raw; imagery still hugs climatology by rarely predicting RI.</em></p>

### Case study (fusion)

<p align="center">
  <img src="assets/cases_fusion_default.png" alt="Fusion hit, miss, and false alarm cases" width="900"/>
</p>

<p align="center"><em>Fusion hit / miss / false alarm on held-out IRWIN scenes (temperature primary).</em></p>

## Research gap / paper direction

Many RI deep-learning reports emphasize that imagery or CNNs “improve” prediction. Fewer ask:

1. Does imagery add skill **beyond SHIPS-like predictors**?
2. Does that improvement survive a **climatology** reference and **calibration** audit?
3. What failure modes appear (over-firing vs under-firing), and do they differ by **basin**?

This project targets that gap. Current answer in one line:

> Imagery helps most for Cat 1–2 storms (only regime with positive imagery BSS); tabular over-fires everywhere, worst on majors and AL; fusion inherits the Cat 1–2 correction and EP skill while keeping usable POD. No temperature-scaled model beats climatology overall, and isotonic “skill” can erase detection.

A defensible paper angle:

> When storm-centered satellite imagery helps (and fails) for probabilistic rapid-intensification prediction: a TC PRIMED ablation study.

## Repository structure

```text
hurricane-ri-forecast/
├── configs/
├── data/
│   ├── raw/tcprimed/        # large, not committed
│   ├── interim/             # prepare checkpoints, not committed
│   └── processed/           # tensors/tables, not committed
├── notebooks/
├── outputs/
│   ├── checkpoints/         # weights, not committed
│   ├── metrics/             # eval/ablation json (can commit)
│   └── figures/             # case/reliability plots (can commit)
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   ├── evaluate.py
│   └── compare_models.py
└── src/
    ├── data/
    ├── models/
    ├── train/
    └── eval/
```

## Setup

```bash
git clone <your-repo-url> hurricane-ri-forecast
cd hurricane-ri-forecast
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

GPU is recommended for the full experiment.

## Pipeline

### 1. Download

```bash
python scripts/download_data.py --config configs/default.yaml --season 2018 --basin AL
```

Interrupted downloads can be resumed; existing files are skipped.

### 2. Prepare

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

Parallel prepare on two machines/Colab sessions:

```bash
python scripts/prepare_data.py --config configs/default.yaml --shard 0/2
python scripts/prepare_data.py --config configs/default.yaml --shard 1/2
python scripts/prepare_data.py --config configs/default.yaml --assemble-only
```

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml
```

### 4. Evaluate and compare

```bash
python scripts/evaluate.py --config configs/default.yaml --model tabular
python scripts/evaluate.py --config configs/default.yaml --model imagery
python scripts/evaluate.py --config configs/default.yaml --model fusion
python scripts/compare_models.py --config configs/default.yaml
```

## Single-storm sanity check

```bash
python scripts/prepare_data.py --config configs/al062018.yaml
python scripts/train.py --config configs/al062018.yaml
python scripts/evaluate.py --config configs/al062018.yaml --model fusion
```

## Google Colab notes

```python
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/hurricane-ri-forecast')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
!pip -q install -r requirements.txt
```

If Drive drops (`Transport endpoint is not connected`), remount and rerun the interrupted command. Prepare resumes from storm checkpoints.

## Evaluation notes

- Split by **season**, never by random frames from the same storm.
- Report probabilistic skill, not accuracy.
- TC PRIMED environmental fields include reanalysis diagnostics: research setting, not claimed real-time operations.
- Optional SHIPS-RII CSV comparison is supported, but beating SHIPS-RII is not the project goal.

```bash
python scripts/evaluate.py --config configs/default.yaml --model fusion --ships-csv data/raw/ships_rii/probabilities.csv
```

CSV columns: `storm_id,time,ships_rii_prob`

## Configs

| File | Purpose |
|---|---|
| `configs/default.yaml` | Full AL/EP multi-season experiment |
| `configs/al062018.yaml` | Single-storm subset |

## Next steps

- Short paper/methods write-up from the README narrative
- Optional SHIPS-RII side-by-side if a matched CSV is available

## Attribution

TC PRIMED is distributed by NOAA. Cite TC PRIMED and follow NOAA open-data attribution guidance when using this project or the underlying data.
