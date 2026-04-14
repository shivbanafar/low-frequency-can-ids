#!/usr/bin/env python3
"""
Cross-dataset evaluation: load a model trained on one dataset (e.g. HCRL) and
evaluate it on a test set generated from a different dataset (e.g. ROAD).

This tests whether the 29x29 CAN-ID representation generalises across vehicles.
No retraining — the model checkpoint is frozen.

Steps this script runs:
  1. Preprocess the cross-dataset CSV into TFRecords  (uses preprocessing_core)
  2. Split into test only (no train/val needed)
  3. Load the trained model checkpoint
  4. Evaluate on the new test set
  5. Print and save metrics to cross_dataset_results.json

Usage:
    # Generate cross-dataset CSV first (e.g. ROAD data):
    python3 generate_dataset.py \
        --normal_csv ../ROAD/ambient_street_driving_long.csv \
        --attack_csvs ../ROAD/accelerator_attack_drive_1.csv \
        --out ../Data_Road/road_lowfreq.csv

    # Then evaluate the HCRL-trained model on it:
    python3 cross_dataset_eval.py \
        --csv        ../Data_Road/road_lowfreq.csv \
        --normal_txt ../ROAD/ambient_street_driving_long.csv \
        --model_path ../Results/all/<hcrl_run_folder> \
        --outdir     ../Data_Road/CrossDataset \
        --out        ../cross_dataset_results.json

Requirements:
    - A trained model checkpoint at --model_path/Saved_models/
    - The cross-dataset CSV in HCRL format (output of generate_dataset.py)
"""
import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

print("Loading TensorFlow...", flush=True)
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import logging
logging.getLogger("tensorflow").disabled = True

from sklearn.metrics import confusion_matrix
from preprocessing_core import preprocess, preprocess_hcrl_normal
from train import Model


def write_tfrecord(data, filename):
    """Write DataFrame (features, label) to TFRecord using native TF (avoids tfrecord pkg conflict)."""
    from tqdm import tqdm
    writer = tf.io.TFRecordWriter(filename)
    for _, row in tqdm(data.iterrows()):
        x = np.asarray(row["features"], dtype=np.int64).flatten().tolist()
        y = int(row["label"])
        feature = {
            "input_features": tf.train.Feature(int64_list=tf.train.Int64List(value=x)),
            "label":          tf.train.Feature(int64_list=tf.train.Int64List(value=[y])),
        }
        example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(example.SerializeToString())
    writer.close()


INPUT_DIM = 29 * 29
ATTACK_NAME = "CrossDataset"
NORMAL_NAME = "Normal"


def preprocess_and_write(csv_path, normal_source, outdir, attack_name):
    """Run preprocessing on the cross-dataset CSV → TFRecords in outdir."""
    outdir = Path(outdir)
    atk_dir = outdir / attack_name
    nor_dir = outdir / NORMAL_NAME
    atk_dir.mkdir(parents=True, exist_ok=True)
    nor_dir.mkdir(parents=True, exist_ok=True)

    print("Preprocessing cross-dataset CSV...", flush=True)

    # Attack windows — pass file path directly (preprocess() reads CSV internally)
    atk_windows = preprocess(csv_path)
    print(f"  Attack windows : {len(atk_windows):,}", flush=True)

    # Normal windows — use dedicated normal source if provided
    if normal_source and Path(normal_source).exists():
        if normal_source.endswith(".txt"):
            nor_windows = preprocess_hcrl_normal(normal_source)
        else:
            nor_windows = preprocess(normal_source)
    else:
        from preprocessing_core import preprocess_r_only_windows
        nor_windows = preprocess_r_only_windows(csv_path)
    print(f"  Normal windows : {len(nor_windows):,}", flush=True)

    # Write full dataset as "test" split (all data goes to test)
    write_tfrecord(atk_windows, str(atk_dir / "test"))
    write_tfrecord(nor_windows, str(nor_dir / "test"))

    # Write datainfo.txt for each class
    for d, w in [(atk_dir, atk_windows), (nor_dir, nor_windows)]:
        info = {"train_unlabel": 0, "train_label": 0, "validation": 0, "test": len(w)}
        with open(d / "datainfo.txt", "w") as f:
            json.dump(info, f)

    return len(atk_windows), len(nor_windows)


def evaluate_on_tfrecords(model_path, data_dir, attack_name, batch_size=64):
    """Load model checkpoint and run inference on the test TFRecords."""
    tf.reset_default_graph()

    model = Model(
        model="CAAE",
        data_dir=data_dir,
        batch_size=batch_size,
        labels=[attack_name, NORMAL_NAME],
    )
    _, y_pred, y_true = model.test(model_path, unknown_test=False)
    return y_true, y_pred


def compute_metrics(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2*precision*recall / (precision+recall) if (precision+recall) > 0 else 0.0
    fnr       = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "precision": round(precision,4), "recall": round(recall,4),
            "f1": round(f1,4), "fnr": round(fnr,4)}


def main():
    parser = argparse.ArgumentParser(description="Cross-dataset evaluation")
    parser.add_argument("--csv",         required=True,
                        help="Cross-dataset low-frequency CSV (output of generate_dataset.py)")
    parser.add_argument("--normal_txt",  default=None,
                        help="Clean normal trace for the cross-dataset vehicle (.txt or .csv). "
                             "Recommended — same reason as in preprocess_low_frequency.py")
    parser.add_argument("--model_path",  required=True,
                        help="Path to trained model run folder (contains Saved_models/)")
    parser.add_argument("--outdir",      default="../Data_CrossDataset",
                        help="Where to write cross-dataset TFRecords (default: ../Data_CrossDataset)")
    parser.add_argument("--out",         default="../cross_dataset_results.json",
                        help="Output JSON path for metrics")
    parser.add_argument("--batch_size",  type=int, default=64)
    args = parser.parse_args()

    # Step 1: Preprocess
    n_atk, n_nor = preprocess_and_write(
        args.csv, args.normal_txt, args.outdir, ATTACK_NAME
    )

    # Step 2: Evaluate
    print(f"\nEvaluating model: {args.model_path}", flush=True)
    y_true, y_pred = evaluate_on_tfrecords(
        args.model_path, args.outdir, ATTACK_NAME, args.batch_size
    )

    # Step 3: Metrics
    metrics = compute_metrics(y_true, y_pred)
    metrics["model_path"] = args.model_path
    metrics["csv"] = args.csv
    metrics["n_attack_windows"] = n_atk
    metrics["n_normal_windows"] = n_nor

    print("\n" + "="*50, flush=True)
    print("Cross-Dataset Evaluation Results", flush=True)
    print("="*50, flush=True)
    print(f"  TP={metrics['TP']}  FN={metrics['FN']}", flush=True)
    print(f"  FP={metrics['FP']}  TN={metrics['TN']}", flush=True)
    print(f"  Precision : {metrics['precision']:.4f}", flush=True)
    print(f"  Recall    : {metrics['recall']:.4f}", flush=True)
    print(f"  F1        : {metrics['f1']:.4f}", flush=True)
    print(f"  FNR       : {metrics['fnr']:.4f}", flush=True)

    with open(args.out, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved → {args.out}", flush=True)


if __name__ == "__main__":
    main()
