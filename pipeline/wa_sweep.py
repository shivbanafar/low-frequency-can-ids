#!/usr/bin/env python3
"""
Sweep over attack class weight w_a ∈ {1, 5, 10, 25, 50, 100}.
For each value: train CAAE (50 epochs), run test, capture metrics.
Saves results to wa_sweep_results.json and prints a summary table.

Usage:
    python3 wa_sweep.py [--data_dir DIR] [--epochs N] [--out FILE]
"""
import argparse
import json
import os
import sys
import logging

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='../Data_MergedLowFreq/Train_0.7_Labeled_0.15')
parser.add_argument('--epochs', type=int, default=50)
parser.add_argument('--out', default='./wa_sweep_results.json')
parser.add_argument('--wa_values', nargs='+', type=float, default=[1, 5, 10, 25, 50, 100])
args = parser.parse_args()

LABELS = ['MergedLowFreq', 'Normal']
BATCH_SIZE = 64
RESULTS_DIR = './Results/all'

print("Loading TensorFlow...", flush=True)
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
logging.getLogger('tensorflow').disabled = True
import numpy as np
from sklearn.metrics import confusion_matrix
print("Ready.\n", flush=True)

from train import Model


def compute_metrics(y_true, y_pred, w_a):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "w_a": w_a,
        "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
        "precision": round(precision, 4),
        "recall":    round(recall,    4),
        "f1":        round(f1,        4),
        "fnr":       round(fnr,       4),
    }


all_results = []

for w_a in args.wa_values:
    print(f"\n{'='*60}", flush=True)
    print(f"  w_a = {w_a}  ({args.epochs} epochs)", flush=True)
    print(f"{'='*60}", flush=True)

    # Record existing result dirs before this training run
    existing_dirs = set(os.listdir(RESULTS_DIR)) if os.path.exists(RESULTS_DIR) else set()

    # ---- TRAIN ----
    tf.reset_default_graph()
    train_model = Model(
        model='CAAE',
        data_dir=args.data_dir,
        batch_size=BATCH_SIZE,
        n_epochs=args.epochs,
        labels=LABELS,
        attack_class_weight=float(w_a),
    )
    train_model.train()

    # Find the new results directory created by this run
    new_dirs = set(os.listdir(RESULTS_DIR)) - existing_dirs
    if not new_dirs:
        print(f"ERROR: could not find new results dir for w_a={w_a}", flush=True)
        continue
    # If somehow multiple new dirs (shouldn't happen), take the most recently modified
    res_dir = max(
        [os.path.join(RESULTS_DIR, d) for d in new_dirs],
        key=os.path.getmtime,
    )
    print(f"\nResults dir: {res_dir}", flush=True)

    # ---- TEST ----
    tf.reset_default_graph()
    test_model = Model(
        model='CAAE',
        data_dir=args.data_dir,
        batch_size=BATCH_SIZE,
        n_epochs=args.epochs,
        labels=LABELS,
        attack_class_weight=float(w_a),
    )
    _, y_pred, y_true = test_model.test(res_dir, unknown_test=False)

    metrics = compute_metrics(y_true, y_pred, w_a)
    all_results.append(metrics)

    print(f"\n  w_a={w_a:>5}  F1={metrics['f1']:.4f}  "
          f"Recall={metrics['recall']:.4f}  FNR={metrics['fnr']:.4f}  "
          f"FP={metrics['FP']}  FN={metrics['FN']}", flush=True)

# ---- SUMMARY TABLE ----
print(f"\n\n{'='*60}")
print("Sweep Summary")
print(f"{'='*60}")
print(f"{'w_a':>6}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'FNR':>7}  {'FP':>5}  {'FN':>5}")
print("-" * 58)
for r in all_results:
    print(f"{r['w_a']:>6}  {r['precision']:>7.4f}  {r['recall']:>7.4f}  "
          f"{r['f1']:>7.4f}  {r['fnr']:>7.4f}  {r['FP']:>5}  {r['FN']:>5}")

# ---- SAVE ----
with open(args.out, 'w') as f:
    json.dump(all_results, f, indent=2)
print(f"\nResults saved to {args.out}")

# ---- PLOT ----
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    wa   = [r['w_a']      for r in all_results]
    f1s  = [r['f1']       for r in all_results]
    recs = [r['recall']   for r in all_results]
    fnrs = [r['fnr']      for r in all_results]
    fps  = [r['FP']       for r in all_results]
    fns  = [r['FN']       for r in all_results]

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    fig.suptitle(r'CAAE performance vs attack class weight $w_a$', fontsize=13)

    axes[0, 0].plot(wa, f1s, 'o-', color='steelblue', lw=2)
    axes[0, 0].set_ylabel('F1 Score'); axes[0, 0].set_title('F1 Score')
    axes[0, 0].set_ylim(0.85, 1.01); axes[0, 0].grid(True, alpha=0.4)

    axes[0, 1].plot(wa, recs, 'o-', color='seagreen', lw=2)
    axes[0, 1].set_ylabel('Recall'); axes[0, 1].set_title('Recall (attack class)')
    axes[0, 1].set_ylim(0.85, 1.01); axes[0, 1].grid(True, alpha=0.4)

    axes[1, 0].plot(wa, fnrs, 'o-', color='firebrick', lw=2)
    axes[1, 0].set_ylabel('FNR'); axes[1, 0].set_title('False Negative Rate')
    axes[1, 0].set_xlabel(r'$w_a$'); axes[1, 0].grid(True, alpha=0.4)

    axes[1, 1].plot(wa, fps, 's--', color='darkorange', lw=2, label='FP')
    axes[1, 1].plot(wa, fns, 'o-',  color='firebrick',  lw=2, label='FN')
    axes[1, 1].set_ylabel('Count'); axes[1, 1].set_title('FP vs FN counts')
    axes[1, 1].set_xlabel(r'$w_a$'); axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.4)

    for ax in axes.flat:
        ax.set_xscale('log')
        ax.set_xticks(wa)
        ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())

    plt.tight_layout()
    plot_path = args.out.replace('.json', '_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to {plot_path}")
except Exception as e:
    print(f"Plot skipped: {e}")
