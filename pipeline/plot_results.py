#!/usr/bin/env python3
"""
Plot validation accuracy and F1 from a training run's log/sum_val.txt.
If sum_val.txt contains four JSON blobs (val acc, val F1, train acc, train F1),
also plots labeled-train vs validation for comparison.

Usage: python plot_results.py [path to run folder]
Example: python plot_results.py "./Results/all/CNN_WGAN_2026-03-14 16:49:50.880337_10_0.0001_64_100_0.5"
"""
import json
import os
import sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def main():
    if len(sys.argv) < 2:
        # default: most recent run in Results/all
        base = "Results/all"
        if not os.path.isdir(base):
            print("Usage: python plot_results.py <path_to_run_folder>")
            print("Example: python plot_results.py './Results/all/CNN_WGAN_2026-03-14 16:49:50.880337_10_0.0001_64_100_0.5'")
            return
        dirs = [os.path.join(base, d) for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))]
        if not dirs:
            print("No run folders found in Results/all")
            return
        run_dir = max(dirs, key=os.path.getmtime)
    else:
        run_dir = sys.argv[1].rstrip("/")

    sum_file = os.path.join(run_dir, "log", "sum_val.txt")
    if not os.path.isfile(sum_file):
        print("Not found:", sum_file)
        return

    with open(sum_file) as f:
        text = f.read()
    # Two JSON objects (older runs): val accs then val f1s; four objects (newer): + train accs, train f1s
    dec = json.JSONDecoder()
    accs, end1 = dec.raw_decode(text)
    rest = text[end1:].lstrip()
    f1s, end2 = dec.raw_decode(rest)
    rest2 = rest[end2:].lstrip()
    accs_train, f1s_train = None, None
    if rest2:
        accs_train, end3 = dec.raw_decode(rest2)
        rest3 = rest2[end3:].lstrip()
        if rest3:
            f1s_train, _ = dec.raw_decode(rest3)

    epochs = list(range(2, 2 * (len(accs["known"])) + 1, 2))  # validation every 2 epochs
    acc = accs["known"]
    f1 = f1s["known"]

    has_train = (
        accs_train is not None
        and f1s_train is not None
        and len(accs_train.get("known", [])) == len(acc)
    )

    if has_train:
        acc_t = accs_train["known"]
        f1_t = f1s_train["known"]
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
        ax1.plot(epochs, acc_t, "C1-o", markersize=3, label="Labeled train (eval pass)")
        ax1.plot(epochs, acc, "b-o", markersize=3, label="Validation")
        ax1.set_ylabel("Accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc="lower right", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, f1_t, "C1-s", markersize=3, label="Labeled train (eval pass)")
        ax2.plot(epochs, f1, "g-s", markersize=3, label="Validation")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("F1")
        ax2.set_ylim(0, 1.05)
        ax2.legend(loc="lower right", fontsize=8)
        ax2.grid(True, alpha=0.3)
        plt.suptitle("CAAE training – train vs validation metrics")
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
        ax1.plot(epochs, acc, "b-o", markersize=3)
        ax1.set_ylabel("Validation accuracy")
        ax1.set_ylim(0, 1.05)
        ax1.grid(True, alpha=0.3)
        ax2.plot(epochs, f1, "g-s", markersize=3)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation F1")
        ax2.set_ylim(0, 1.05)
        ax2.grid(True, alpha=0.3)
        plt.suptitle("CAAE training – validation metrics")
    plt.tight_layout()
    out = os.path.join(run_dir, "log", "validation_curves.png")
    plt.savefig(out, dpi=150)
    print("Saved:", out)

if __name__ == "__main__":
    main()
