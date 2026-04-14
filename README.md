# CAN Bus IDS — Low-Frequency Attack Detection

> **Paper:** "Catching Rare Injections: Semi-Supervised CAAE with Class-Weighted Loss  
> for Low-Frequency CAN Intrusion Detection"  
> Shiv Banafar, Mohammed Afzal Asim — NSUT, New Delhi

---

## What This Is

Modern CAN bus IDS research almost always evaluates on datasets where attack traffic is
10–50% of all frames, making detection easy. This project asks: **what happens at 0.5%
injection rate?** At that density a frequency-counting monitor sees nothing anomalous.

We build a low-frequency benchmark by injecting real HCRL/ROAD attack frames into clean
normal traces at a controlled 0.5% rate, then train a **semi-supervised Convolutional
Adversarial Autoencoder (CAAE)** on 29×29 CAN-ID binary windows. The key addition is
**class-weighted cross-entropy** in the supervised head to counter the resulting 4:1
normal-to-attack imbalance.

A sweep over attack class weight `w_a ∈ {1, 5, 10, 25, 50, 100}` shows:
- F1 is stable and high across `w_a ∈ [5, 25]` (all > 0.983)
- FNR decreases monotonically with `w_a`
- Precision collapses sharply at `w_a ≥ 50` (model flags everything as attack)
- The CAAE with any `w_a ∈ [5, 25]` outperforms all supervised baselines

---

## Key Results (held-out test set)

| Method | Precision | Recall | F1 | FNR |
|---|---|---|---|---|
| Statistical Profile (z-score) | 1.000 | 0.931 | 0.964 | 0.069 |
| Decision Tree | 0.810 | 0.632 | 0.710 | 0.368 |
| Random Forest | 1.000 | 0.567 | 0.724 | 0.433 |
| SVM (RBF, C=10) | 1.000 | 0.954 | 0.976 | 0.046 |
| MLP (1000, 1000) | 1.000 | 0.961 | 0.980 | 0.039 |
| CAAE w_a=1 (unweighted) | 0.998 | 0.960 | 0.978 | 0.040 |
| **CAAE w_a=10 (F1-optimal)** | **0.986** | **0.984** | **0.985** | **0.016** |
| CAAE w_a=25 (low-FNR) | 0.949 | 0.988 | 0.968 | 0.012 |

---

## Repository Layout

```
pipeline/                   <- everything needed to replicate the paper
│
├── CAAE.py                  Model architecture: encoder, decoder, two discriminators
├── AAE.py                   Alternative AAE architecture (used by train.py --model AAE)
├── cnn.py                   CNN building blocks (conv layers, pooling) used by CAAE
├── utils.py                 TFRecord loading, evaluation metrics, results folder naming
├── tfrecord_utils.py        TFRecord writing without TF dependency (used by preprocessing)
│
├── preprocessing_core.py    Core 29x29 windowing engine — shared by both preprocess scripts
├── preprocess_low_frequency.py  Step 1: CSV + normal_run_data.txt -> TFRecords
├── train_test_split.py      Step 2: TFRecords -> 70/15/15 train-val-test split
│
├── train.py                 Step 3: trains and evaluates the CAAE/AAE
├── plot_results.py          Step 4: reads training logs -> validation curve PNG
│
├── compare_baselines.py     Step 5a: sklearn baselines (DT, RF, SVM, MLP)
├── frequency_detector.py    Step 5b: z-score bit-profile anomaly detector
├── wa_sweep.py              Step 5c: trains CAAE across w_a values, saves sweep results
│
├── generate_dataset.py      Step 0: raw CSV + normal trace -> low-frequency CSV at configurable injection rate\n├── cross_dataset_eval.py    Step 6: evaluate HCRL-trained model on a different vehicle's data\n│\n└── requirements.txt         Python dependencies

archive/                    <- not needed for this paper
│                              (original HCRL pipeline, notebooks, dev scripts)
├── notebooks/               Original repo Jupyter notebooks
├── Data/                    Original HCRL TFRecords (standard benchmark splits)
├── preprocessing.py         Old HCRL pipeline — replaced by preprocess_low_frequency.py
├── run_pipeline.sh          Old HCRL shell pipeline
├── run_low_frequency_pipeline.sh  Outdated — uses wrong preprocessing script
├── test_performance.py      Inference timing benchmark
├── test_imports.py          Dev debugging helper
├── text_to_csv.py           One-time HCRL .txt -> CSV converter (standalone)
└── wa_sweep.log             Full training log from the paper's sweep run

DATASET/                    <- raw source data (needed only for Step 1)
├── DoS_dataset.csv
├── Fuzzy_dataset.csv
├── RPM_dataset.csv
├── gear_dataset.csv
└── normal_run_data.txt      Clean normal trace used for normal window extraction

Data_MergedLowFreq/         <- preprocessed TFRecords (output of Steps 1-2)
└── Train_0.7_Labeled_0.15/
    ├── MergedLowFreq/       Attack windows: train_label, train_unlabel, val, test
    └── Normal/              Normal windows: train_label, train_unlabel, val, test

Results/                    <- trained model checkpoints (output of Step 3)
└── all/
    └── CNN_WGAN_<timestamp>_10_0.0001_64_50_0.5/
        ├── Saved_models/    Checkpoint files (restored by train.py for evaluation)
        ├── Tensorboard/     TensorBoard event files
        └── log/
            ├── sum_val.txt  Per-epoch validation metrics (read by plot_results.py)
            └── log.txt      Training log

baseline_results.json        Sklearn baseline metrics — Table 3 in paper
frequency_results.json       z-score detector metrics — Table 3 in paper
wa_sweep_results.json        Per-w_a metrics — Table 2 in paper
wa_sweep_results_plot.png    FP/FN trade-off plot — Fig. 2 in paper
```

---

## Architecture: CAAE

```
Input (29x29)
    |
[Encoder: 4x Conv → FC]
    |           |
  z (dim=10)  ŷ (2-way logit)
    |           |         |
[Gaussian   [Categorical  [Supervised loss]  <- labeled examples only
  Disc.]      Disc.]       w(0)=1, w(1)=w_a
    |
[Decoder: FC → 4x Deconv] -> x̂
    |
  MSE loss
```

The encoder is trained adversarially (WGAN-GP) to push `z` toward N(0,5I) and `ŷ`
toward a uniform categorical prior. All three losses (reconstruction, adversarial,
supervised) shape the same encoder simultaneously. Class weighting only affects the
supervised loss — no architectural changes.

---

## Data Split

| Split | Normal | Attack | Labeled? |
|---|---|---|---|
| Train (labeled) | 3,580 | 906 | Yes |
| Train (unlabeled) | 20,335 | 5,093 | No |
| Validation | 5,167 | 1,242 | Yes |
| Test | 5,104 | 1,305 | Yes |
| **Total** | **34,186** | **8,546** | — |

The labeled split is 15% of training data. The CAAE uses both labeled and unlabeled
windows during training; supervised baselines only see the labeled portion.

---

## Quick Start

```bash
# 1. Install dependencies
python3 -m venv venv && source venv/bin/activate
pip install -r pipeline/requirements.txt

# 2. All commands run from pipeline/
cd pipeline

# 3. Run baselines on pre-processed data (no training needed)
python3 compare_baselines.py
python3 frequency_detector.py

# 4. Train and evaluate CAAE
python3 train.py \
    --data_dir ../Data_MergedLowFreq/Train_0.7_Labeled_0.15 \
    --labels MergedLowFreq Normal --epochs 50 \
    --attack_class_weight 10 --is_train

# 5. Reproduce the class-weight sweep
python3 wa_sweep.py
```

---

## Dependencies

- Python 3.8+
- TensorFlow 1.x (via `tensorflow-cpu` or `tensorflow.compat.v1`)
- scikit-learn, numpy, matplotlib, tqdm, tfrecord

See `pipeline/requirements.txt` for pinned versions.

---

## Citation

Built on the CAAE architecture from:
> Ngo et al., "Detecting In-vehicle Intrusion via Semi-supervised Learning-based
> Convolutional Adversarial Autoencoders", Vehicle Communications, 2022.
> https://doi.org/10.1016/j.vehcom.2022.100520

Dataset sources:
- HCRL Car-Hacking Dataset: https://ocslab.hksecurity.net/Datasets/CAN-intrusion-dataset
