#!/usr/bin/env python3
"""
Frequency-based IDS baseline (adapted from Young et al. 2019).

For each 29x29 window, count how often each bit position is set (sum over rows → 29-dim
per-row profile, or flat 841-dim).  We use the flat 841-dim vector (each cell = sum of
that bit across 29 frames in the window).

Baseline profile = mean per-cell count over normal training windows.
Detection rule   = z-score distance from baseline profile.
                   Flag as attack if distance > threshold.

Sweep threshold on validation/test to find best-F1 operating point.

Usage:
    python3 frequency_detector.py [--data_root DATA_ROOT]
"""
import argparse
import json
import os
import sys

print("Loading TensorFlow...", flush=True)
import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print("TensorFlow loaded.", flush=True)

INPUT_DIM = 29 * 29  # 841


def read_tfrecord_fn(example):
    feature_description = {
        "input_features": tf.io.FixedLenFeature([INPUT_DIM], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)


def load_tfrecords_to_numpy(paths):
    paths = [p for p in paths if os.path.isfile(p)]
    if not paths:
        return np.zeros((0, INPUT_DIM), dtype=np.float32), np.zeros((0,), dtype=np.int64)

    dataset = tf.data.Dataset.from_tensor_slices(paths)
    dataset = dataset.interleave(
        lambda x: tf.data.TFRecordDataset(x),
        cycle_length=len(paths),
        block_length=10000,
    )
    dataset = dataset.map(read_tfrecord_fn, num_parallel_calls=4)
    dataset = dataset.batch(10000)
    iterator = dataset.make_one_shot_iterator()
    next_batch = iterator.get_next()

    X_parts, y_parts = [], []
    with tf.Session() as sess:
        while True:
            try:
                batch = sess.run(next_batch)
                X_parts.append(batch["input_features"].astype(np.float32))
                y_parts.append(batch["label"].flatten())
            except tf.errors.OutOfRangeError:
                break

    X = np.concatenate(X_parts, axis=0) if X_parts else np.zeros((0, INPUT_DIM), dtype=np.float32)
    y = np.concatenate(y_parts, axis=0) if y_parts else np.zeros((0,), dtype=np.int64)
    return X, y


def metrics_at_threshold(scores, y_true, threshold):
    y_pred = (scores > threshold).astype(int)
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fnr": round(fnr, 4),
            "threshold": float(threshold)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../Data_MergedLowFreq/Train_0.7_Labeled_0.15")
    parser.add_argument("--out", default="./frequency_results.json")
    args = parser.parse_args()
    root = args.data_root.rstrip("/")

    # Load normal training windows (labeled + unlabeled) to build profile
    normal_train_paths = [
        f"{root}/Normal/train_label",
        f"{root}/Normal/train_unlabel",
    ]
    print("Loading normal training windows...", flush=True)
    X_normal_train, _ = load_tfrecords_to_numpy(normal_train_paths)
    print(f"  Normal train windows: {X_normal_train.shape[0]}", flush=True)

    # Build baseline profile: mean + std per cell
    profile_mean = X_normal_train.mean(axis=0)        # (841,)
    profile_std  = X_normal_train.std(axis=0) + 1e-8  # (841,) — avoid div/0

    # Load test data (attack + normal)
    test_paths_attack = [f"{root}/MergedLowFreq/test"]
    test_paths_normal = [f"{root}/Normal/test"]
    print("Loading test data...", flush=True)
    X_atk_te, y_atk_te = load_tfrecords_to_numpy(test_paths_attack)
    X_nor_te, y_nor_te = load_tfrecords_to_numpy(test_paths_normal)
    X_test = np.concatenate([X_atk_te, X_nor_te], axis=0)
    y_test = np.concatenate([y_atk_te, y_nor_te], axis=0)
    print(f"  Test: {X_test.shape[0]} samples  (attack={int((y_test==1).sum())}  normal={int((y_test==0).sum())})", flush=True)

    # Compute anomaly score: mean absolute z-score across all cells
    z = np.abs((X_test - profile_mean) / profile_std)   # (N, 841)
    scores = z.mean(axis=1)                              # (N,) scalar per window

    # Sweep thresholds
    thresholds = np.linspace(0, scores.max(), 200)
    best_f1, best_result = -1, None
    for thr in thresholds:
        r = metrics_at_threshold(scores, y_test, thr)
        if r["f1"] > best_f1:
            best_f1 = r["f1"]
            best_result = r

    # Also report at mean+3std of normal scores (classic z-score rule)
    normal_scores = scores[y_test == 0]
    fixed_threshold = normal_scores.mean() + 3 * normal_scores.std()
    fixed_result = metrics_at_threshold(scores, y_test, fixed_threshold)

    print("\n\nFrequency Detector Results")
    print("=" * 50)
    print(f"\nBest-F1 threshold = {best_result['threshold']:.4f}")
    print(f"  TP={best_result['TP']}  FN={best_result['FN']}")
    print(f"  FP={best_result['FP']}  TN={best_result['TN']}")
    print(f"  Precision : {best_result['precision']:.4f}")
    print(f"  Recall    : {best_result['recall']:.4f}")
    print(f"  F1        : {best_result['f1']:.4f}")
    print(f"  FNR       : {best_result['fnr']:.4f}")

    print(f"\nFixed threshold (mean+3std of normal) = {fixed_threshold:.4f}")
    print(f"  TP={fixed_result['TP']}  FN={fixed_result['FN']}")
    print(f"  FP={fixed_result['FP']}  TN={fixed_result['TN']}")
    print(f"  Precision : {fixed_result['precision']:.4f}")
    print(f"  Recall    : {fixed_result['recall']:.4f}")
    print(f"  F1        : {fixed_result['f1']:.4f}")
    print(f"  FNR       : {fixed_result['fnr']:.4f}")

    output = {
        "best_f1_threshold": best_result,
        "fixed_threshold_mean_plus_3std": fixed_result,
    }
    with open(args.out, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
