#!/usr/bin/env python3
"""
Generate a low-frequency CAN attack dataset by injecting attack frames
into a clean normal trace at a configurable rate.

Supports two input formats:
  - HCRL CSV:  Timestamp,CAN_ID,DLC,D0-D7,Flag  (Flag='R' normal, 'T' attack)
  - HCRL .txt: Timestamp: X  ID: Y  000  DLC: Z  D0 D1 ...  (normal_run_data.txt)

Output: CSV in HCRL format — same columns, Flag='R' for normal, 'T' for injected.

Usage (HCRL — replicates the paper's dataset):
    python3 generate_dataset.py \
        --normal_txt  ../DATASET/normal_run_data.txt \
        --attack_csvs ../DATASET/DoS_dataset.csv ../DATASET/Fuzzy_dataset.csv \
                      ../DATASET/RPM_dataset.csv ../DATASET/gear_dataset.csv \
        --injection_rate 0.005 \
        --out ../Data_MergedLowFreq/low_frequency_can_dataset.csv

Usage (ROAD or any other vehicle — cross-dataset):
    python3 generate_dataset.py \
        --normal_csv  ../ROAD/ambient_street_driving_long.csv \
        --attack_csvs ../ROAD/accelerator_attack_drive_1.csv \
        --injection_rate 0.005 \
        --out ../Data_Road/low_frequency_road_dataset.csv
"""
import argparse
import csv
import random
import re
import sys
from pathlib import Path


def parse_hcrl_txt(txt_path):
    """Parse HCRL .txt normal trace — returns rows in CSV format with Flag='R'."""
    pattern = re.compile(
        r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\S+\s+DLC:\s*(\d+)\s*(.*)"
    )
    rows = []
    with open(txt_path, "r") as f:
        for line in f:
            m = pattern.match(line.strip())
            if not m:
                continue
            ts, can_id, dlc, data_str = m.group(1), m.group(2), m.group(3), m.group(4)
            data_bytes = (data_str.strip().split() + ["00"] * 8)[:8]
            rows.append([ts, can_id.lower(), dlc] + data_bytes + ["R"])
    return rows


def parse_hcrl_csv(csv_path, flag_filter=None):
    """Parse HCRL CSV. flag_filter='R'/'T'/None to keep all."""
    rows = []
    with open(csv_path, "r") as f:
        for row in csv.reader(f):
            if len(row) < 12:
                continue
            flag = row[11].strip().upper()
            if flag_filter and flag != flag_filter:
                continue
            rows.append(row[:12])
    return rows


def load_attack_pool(attack_csvs):
    pool = []
    for path in attack_csvs:
        rows = parse_hcrl_csv(path, flag_filter="T")
        if not rows:
            # All-R attack capture (some HCRL files) — treat every row as attack
            rows = parse_hcrl_csv(path)
            for r in rows:
                r[11] = "T"
        pool.extend(rows)
        print(f"  {len(rows):>9,} attack frames  ← {Path(path).name}", flush=True)
    return pool


def inject(normal_rows, attack_pool, injection_rate, seed):
    rng = random.Random(seed)
    out = []
    for row in normal_rows:
        if rng.random() < injection_rate:
            atk = list(rng.choice(attack_pool))
            try:
                atk[0] = f"{float(row[0]) - 0.0001:.6f}"
            except (ValueError, IndexError):
                pass
            atk[11] = "T"
            out.append(atk)
        out.append(row)
    return out


def main():
    parser = argparse.ArgumentParser(description="Generate low-frequency CAN attack dataset")
    parser.add_argument("--normal_csv",  type=str, default=None,
                        help="Normal trace in HCRL CSV format")
    parser.add_argument("--normal_txt",  type=str, default=None,
                        help="Normal trace in HCRL .txt format (e.g. normal_run_data.txt)")
    parser.add_argument("--attack_csvs", nargs="+", required=True,
                        help="One or more attack CSVs (HCRL format)")
    parser.add_argument("--injection_rate", type=float, default=0.005,
                        help="Injection probability per frame (default: 0.005 = 0.5%%)")
    parser.add_argument("--out", type=str,
                        default="../Data_MergedLowFreq/low_frequency_can_dataset.csv",
                        help="Output CSV path")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    if not args.normal_csv and not args.normal_txt:
        print("ERROR: provide --normal_csv or --normal_txt", file=sys.stderr)
        sys.exit(1)

    print("Loading normal trace...", flush=True)
    normal_rows = parse_hcrl_txt(args.normal_txt) if args.normal_txt \
                  else parse_hcrl_csv(args.normal_csv, flag_filter="R") or \
                       parse_hcrl_csv(args.normal_csv)
    print(f"  {len(normal_rows):,} normal frames", flush=True)

    print("Loading attack pool...", flush=True)
    attack_pool = load_attack_pool(args.attack_csvs)
    print(f"  {len(attack_pool):,} attack frames total", flush=True)

    if not attack_pool:
        print("ERROR: no attack frames found", file=sys.stderr)
        sys.exit(1)

    print(f"Injecting at {args.injection_rate*100:.2f}% rate (seed={args.seed})...", flush=True)
    merged = inject(normal_rows, attack_pool, args.injection_rate, args.seed)

    n_atk = sum(1 for r in merged if r[11].strip().upper() == "T")
    n_nor = len(merged) - n_atk
    print(f"  Total  : {len(merged):,}", flush=True)
    print(f"  Normal : {n_nor:,}", flush=True)
    print(f"  Attack : {n_atk:,}  ({n_atk/len(merged)*100:.3f}%)", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerows(merged)
    print(f"Saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
