# hurricane-ri-forecast

Deep learning for **24-hour hurricane rapid intensification (RI)** using NOAA storm-centered satellite infrared imagery and environmental fields.

Rapid intensification is defined here as a **≥ 30 kt** increase in maximum sustained wind over **24 hours**. The project asks a specific question:

> Does raw satellite imagery add RI forecast skill beyond environmental and infrared-summary predictors similar to those used by operational aids such as **SHIPS-RII**?

Models are compared with a season-held-out split and probabilistic metrics (not accuracy).

---

## Features

- End-to-end pipeline: download → prepare → train → evaluate
- NOAA **TC PRIMED** ingestion (NetCDF overpasses + environmental diagnostics)
- RI labels from best-track intensity change at +24 h
- Three-model ablation:
  - `tabular`: environment + IR summary features
  - `imagery`: CNN on storm-centered `IRWIN` fields
  - `fusion`: tabular + imagery
- Season-based train / val / test splits (Atlantic + eastern Pacific by default)
- Evaluation with Brier score, Brier skill vs climatology, bootstrap intervals, reliability diagrams
- Case review figures: hit, miss, and false alarm
- Storm-level prepare checkpoints so long runs can resume after interruption
- Optional hook for official SHIPS-RII probabilities (`--ships-csv`)

---

## Dataset

Primary data source: **[TC PRIMED](https://rammb-data.cira.colostate.edu/tcprimed/)** (NOAA / CIRA), public on AWS:

`s3://noaa-nesdis-tcprimed-pds`

Default experiment scope (`configs/default.yaml`):

| Split | Seasons | Basins |
|---|---|---|
| Train / val | 2017–2022 | AL, EP |
| Test | 2023–2024 | AL, EP |

A smaller config (`configs/al062018.yaml`) runs a single-storm sanity check on `AL062018`.

Place raw files under:

```text
data/raw/tcprimed/<year>/<basin>/<storm_id>/
```

---

## Repository structure

```text
hurricane-ri-forecast/
├── configs/                 # experiment configs
├── data/
│   ├── raw/tcprimed/        # downloaded NetCDF storms
│   ├── interim/             # prepare checkpoints
│   └── processed/           # tensors + sample tables
├── notebooks/               # optional Colab notebooks
├── outputs/
│   ├── checkpoints/
│   ├── metrics/
│   └── figures/
├── scripts/
│   ├── download_data.py
│   ├── prepare_data.py
│   ├── train.py
│   └── evaluate.py
└── src/
    ├── data/                # TC PRIMED IO, labels, datasets
    ├── models/              # tabular, imagery, fusion
    ├── train/               # training loop + metrics
    └── eval/                # cases + optional SHIPS-RII compare
```

---

## Setup

```bash
git clone <your-repo-url> hurricane-ri-forecast
cd hurricane-ri-forecast
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

GPU is recommended for training. CPU works for small configs.

---

## Quick start

### 1. Download data

One storm:

```bash
python scripts/download_data.py --config configs/default.yaml --season 2018 --basin AL --storm 06
```

One basin-season:

```bash
python scripts/download_data.py --config configs/default.yaml --season 2018 --basin AL
```

Already-downloaded files are skipped, so interrupted downloads can be resumed by rerunning the same command.

### 2. Prepare samples

```bash
python scripts/prepare_data.py --config configs/default.yaml
```

This builds labeled samples, writes storm-level checkpoints to `data/interim/prepare_default/`, and saves the final dataset to `data/processed/`.

If the run stops early, rerun the same command. Finished storms are skipped.

### 3. Train

```bash
python scripts/train.py --config configs/default.yaml
```

Trains `tabular`, `imagery`, and `fusion` (configurable in YAML).

### 4. Evaluate

```bash
python scripts/evaluate.py --config configs/default.yaml --model tabular
python scripts/evaluate.py --config configs/default.yaml --model imagery
python scripts/evaluate.py --config configs/default.yaml --model fusion
```

Outputs land in:

- `outputs/metrics/eval_<model>_default.json`
- `outputs/figures/cases_<model>_default.png`
- `outputs/figures/reliability_<model>_default.png`

Optional official SHIPS-RII comparison:

```bash
python scripts/evaluate.py --config configs/default.yaml --model fusion --ships-csv data/raw/ships_rii/probabilities.csv
```

CSV columns: `storm_id,time,ships_rii_prob`

---

## Single-storm check

Useful before a full multi-season run:

```bash
python scripts/prepare_data.py --config configs/al062018.yaml
python scripts/train.py --config configs/al062018.yaml
python scripts/evaluate.py --config configs/al062018.yaml --model fusion
```

---

## Google Colab

1. Put the repo on Google Drive (or clone into Drive).
2. Mount Drive in Colab.
3. `cd` into the project root.
4. Install requirements and run the same `scripts/` commands.

Example:

```python
from google.colab import drive
import os

drive.mount('/content/drive', force_remount=True)
os.chdir('/content/drive/MyDrive/hurricane-ri-forecast')
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

!pip -q install -r requirements.txt
!python scripts/prepare_data.py --config configs/default.yaml
```

If Drive disconnects (`Transport endpoint is not connected`), remount with `force_remount=True`, `cd` back into the project, and rerun the interrupted script.

---

## Evaluation notes

- Splits are by **season**, not by random frames from the same storm.
- RI is rare, so headline metrics are **Brier skill** and reliability, plus POD / success ratio at a chosen threshold.
- Environmental fields from TC PRIMED include reanalysis-based diagnostics. Treat this as a **research setting**, not a claim of operational real-time forecast inputs.
- Without attached official SHIPS-RII probabilities, the ablation still answers whether imagery adds skill beyond SHIPS-like tabular predictors built from TC PRIMED.

---

## Configs

| File | Purpose |
|---|---|
| `configs/default.yaml` | Multi-season AL/EP experiment |
| `configs/al062018.yaml` | Single-storm subset (`AL062018`, GMI-focused) |

Edit seasons, basins, batch size, epochs, and model list in these files as needed.

---

## License / attribution

TC PRIMED is distributed by NOAA. If you use this project or the underlying data, cite TC PRIMED and follow NOAA open-data attribution guidance.
