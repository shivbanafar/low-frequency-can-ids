#!/usr/bin/env python3
"""
Baseline classifiers for CAN bus IDS comparison.
Loads labeled-train + test TFRecords, trains sklearn models, prints and saves metrics.

Usage:
    python3 compare_baselines.py [--data_root DATA_ROOT]

Data root defaults to ../Data_MergedLowFreq/Train_0.7_Labeled_0.15
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

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix

INPUT_DIM = 29 * 29  # 841


def read_tfrecord_fn(example):
    feature_description = {
        "input_features": tf.io.FixedLenFeature([INPUT_DIM], tf.int64),
        "label": tf.io.FixedLenFeature([1], tf.int64),
    }
    return tf.io.parse_single_example(example, feature_description)


def load_tfrecords_to_numpy(paths):
    """Load one or more TFRecord files into (X, y) numpy arrays."""
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


def compute_metrics(y_true, y_pred, name):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    total = tp + tn + fp + fn
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    fnr = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    acc = (tp + tn) / total if total > 0 else 0.0
    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  TP={tp}  FN={fn}")
    print(f"  FP={fp}  TN={tn}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  F1        : {f1:.4f}")
    print(f"  FNR       : {fnr:.4f}")
    print(f"  Accuracy  : {acc:.4f}")
    return {"name": name, "TP": int(tp), "FN": int(fn), "FP": int(fp), "TN": int(tn),
            "precision": round(precision, 4), "recall": round(recall, 4),
            "f1": round(f1, 4), "fnr": round(fnr, 4), "accuracy": round(acc, 4)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="../Data_MergedLowFreq/Train_0.7_Labeled_0.15")
    parser.add_argument("--out", default="./baseline_results.json")
    args = parser.parse_args()

    root = args.data_root.rstrip("/")

    # TFRecord paths
    attack_train = f"{root}/MergedLowFreq/train_label"
    normal_train  = f"{root}/Normal/train_label"
    attack_test   = f"{root}/MergedLowFreq/test"
    normal_test   = f"{root}/Normal/test"

    print("\nLoading train data...", flush=True)
    X_atk_tr, y_atk_tr = load_tfrecords_to_numpy([attack_train])
    X_nor_tr, y_nor_tr = load_tfrecords_to_numpy([normal_train])
    X_train = np.concatenate([X_atk_tr, X_nor_tr], axis=0)
    y_train = np.concatenate([y_atk_tr, y_nor_tr], axis=0)
    print(f"  Train: {X_train.shape[0]} samples  (attack={int((y_train==1).sum())}  normal={int((y_train==0).sum())})", flush=True)

    print("Loading test data...", flush=True)
    X_atk_te, y_atk_te = load_tfrecords_to_numpy([attack_test])
    X_nor_te, y_nor_te = load_tfrecords_to_numpy([normal_test])
    X_test = np.concatenate([X_atk_te, X_nor_te], axis=0)
    y_test = np.concatenate([y_atk_te, y_nor_te], axis=0)
    print(f"  Test : {X_test.shape[0]} samples  (attack={int((y_test==1).sum())}  normal={int((y_test==0).sum())})", flush=True)

    models = [
        ("Decision Tree",   DecisionTreeClassifier(class_weight="balanced", random_state=42)),
        ("Random Forest",   RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=42)),
        ("SVM (RBF, C=10)", SVC(kernel="rbf", C=10, class_weight="balanced", random_state=42)),
        ("MLP (1000,1000)", MLPClassifier(hidden_layer_sizes=(1000, 1000), max_iter=200, random_state=42)),
    ]

    results = []
    for name, clf in models:
        print(f"\nTraining {name}...", flush=True)
        clf.fit(X_train, y_train)
        print(f"Evaluating {name}...", flush=True)
        y_pred = clf.predict(X_test)
        r = compute_metrics(y_test, y_pred, name)
        results.append(r)

    print("\n\nSummary Table")
    print(f"{'Method':<22} {'Prec':>7} {'Rec':>7} {'F1':>7} {'FNR':>7}")
    print("-" * 54)
    for r in results:
        print(f"{r['name']:<22} {r['precision']:>7.4f} {r['recall']:>7.4f} {r['f1']:>7.4f} {r['fnr']:>7.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.out}")


if __name__ == "__main__":
    main()
