"""
CAN frame CSV -> 29x29 windowed features (no TensorFlow).
Used by preprocess_low_frequency.py so heavy pandas work runs before TF loads.
"""
import sys

sys.stdout.reconfigure(line_buffering=True)

import numpy as np
import pandas as pd
from tqdm import tqdm

attributes = [
    "Timestamp",
    "canID",
    "DLC",
    "Data0",
    "Data1",
    "Data2",
    "Data3",
    "Data4",
    "Data5",
    "Data6",
    "Data7",
    "Flag",
]


def convert_canid_bits(cid):
    try:
        s = bin(int(str(cid), 16))[2:].zfill(29)
        bits = list(map(int, list(s)))
        return bits
    except Exception:
        return None


def preprocess(file_name):
    print("[1/5] Reading CSV: {}".format(file_name), flush=True)
    pd_df = pd.read_csv(file_name, header=None, names=attributes, dtype=str)
    total = len(pd_df)
    print("      {} rows loaded".format(total), flush=True)

    print("[2/5] Fixing Flag column for short DLC rows...", flush=True)
    mask = pd_df["Flag"].isna()
    if mask.any():
        for idx in tqdm(pd_df[mask].index, desc="  fix_flag"):
            dlc = pd_df.at[idx, "DLC"]
            pd_df.at[idx, "Flag"] = pd_df.at[idx, "Data" + str(dlc)]
    print("      Fixed {} rows".format(mask.sum()), flush=True)

    pd_df = pd_df[["Timestamp", "canID", "Flag"]].sort_values("Timestamp", ascending=True)

    print("[3/5] Converting CAN IDs to 29-bit vectors...", flush=True)
    tqdm.pandas(desc="  canID->bits")
    pd_df["canBits"] = pd_df["canID"].progress_apply(convert_canid_bits)
    pd_df["Flag"] = pd_df["Flag"].apply(lambda x: True if x == "T" else False)

    print("[4/5] Building 29x29 feature windows...", flush=True)
    as_strided = np.lib.stride_tricks.as_strided
    win = 29
    s = 29
    feature = as_strided(
        pd_df.canBits, ((len(pd_df) - win) // s + 1, win), (8 * s, 8)
    )
    label = as_strided(
        pd_df.Flag, ((len(pd_df) - win) // s + 1, win), (1 * s, 1)
    )
    df = pd.DataFrame(
        {
            "features": pd.Series(feature.tolist()),
            "label": pd.Series(label.tolist()),
        },
        index=range(len(feature)),
    )

    print("[5/5] Assigning labels...", flush=True)
    tqdm.pandas(desc="  labels")
    df["label"] = df["label"].progress_apply(lambda x: 1 if any(x) else 0)

    print("Preprocessing: DONE", flush=True)
    print("#Normal: ", df[df["label"] == 0].shape[0])
    print("#Attack: ", df[df["label"] == 1].shape[0])
    return df[["features", "label"]].reset_index().drop(["index"], axis=1)


def preprocess_hcrl_normal(file_name):
    """
    Build 29x29 windows with label 0 from HCRL normal_run_data.txt.

    This follows the paper's method: use a separate, attack-free normal capture
    for normal windows instead of extracting R-only rows from a mixed attack/normal file.
    Eliminates the domain mismatch caused by preprocess_r_only_windows on mixed CSVs.
    """
    import re
    print("[HCRL-Normal 1/4] Reading: {}".format(file_name), flush=True)
    pat = re.compile(
        r"Timestamp:\s*([\d.]+)\s+ID:\s*([0-9a-fA-F]+)\s+\d+\s+DLC:\s*\d+\s+(.*)"
    )
    can_ids = []
    with open(file_name) as f:
        for line in f:
            m = pat.match(line.strip())
            if m:
                can_ids.append(m.group(2).lower())
    print("      {} rows parsed".format(len(can_ids)), flush=True)
    if len(can_ids) < 29:
        raise ValueError("Need at least 29 rows; got {}".format(len(can_ids)))

    print("[HCRL-Normal 2/4] Converting CAN IDs to 29-bit vectors...", flush=True)
    pd_s = pd.Series(can_ids, name="canID")
    tqdm.pandas(desc="  canID->bits")
    bits = pd_s.progress_apply(convert_canid_bits)
    bits = bits.dropna().reset_index(drop=True)
    if len(bits) < 29:
        raise ValueError("Need at least 29 valid CAN IDs; got {}".format(len(bits)))

    print("[HCRL-Normal 3/4] Building 29x29 windows (all label 0)...", flush=True)
    as_strided = np.lib.stride_tricks.as_strided
    win = 29
    s = 29
    feature = as_strided(bits, ((len(bits) - win) // s + 1, win), (8 * s, 8))
    df = pd.DataFrame(
        {
            "features": pd.Series(feature.tolist()),
            "label": np.zeros(len(feature), dtype=np.int64),
        },
        index=range(len(feature)),
    )
    print("[HCRL-Normal 4/4] DONE — #Normal windows: {}".format(len(df)), flush=True)
    return df[["features", "label"]].reset_index().drop(["index"], axis=1)


def preprocess_r_only_windows(file_name):
    """
    Build 29x29 windows with label 0 from R-only frames (chronological order).
    Use for mixed R/T CSVs where dense T frames yield no all-R windows under preprocess().
    """
    print("[R-only 1/4] Reading CSV: {}".format(file_name), flush=True)
    pd_df = pd.read_csv(file_name, header=None, names=attributes, dtype=str)
    print("      {} rows loaded".format(len(pd_df)), flush=True)

    print("[R-only 2/4] Fixing Flag column...", flush=True)
    mask = pd_df["Flag"].isna()
    if mask.any():
        for idx in tqdm(pd_df[mask].index, desc="  fix_flag"):
            dlc = pd_df.at[idx, "DLC"]
            pd_df.at[idx, "Flag"] = pd_df.at[idx, "Data" + str(dlc)]

    pd_df = pd_df[["Timestamp", "canID", "Flag"]].sort_values("Timestamp", ascending=True)
    pd_df = pd_df[pd_df["Flag"].apply(lambda x: str(x).strip().upper() != "T")].reset_index(
        drop=True
    )
    print("[R-only] R-only rows: {}".format(len(pd_df)), flush=True)
    if len(pd_df) < 29:
        raise ValueError(
            "Need at least 29 R-only frames for windowing; got {}".format(len(pd_df))
        )

    print("[R-only 3/4] Converting CAN IDs to 29-bit vectors...", flush=True)
    tqdm.pandas(desc="  canID->bits")
    pd_df["canBits"] = pd_df["canID"].progress_apply(convert_canid_bits)
    pd_df = pd_df.dropna(subset=["canBits"])
    if len(pd_df) < 29:
        raise ValueError(
            "Need at least 29 R-only rows after CAN ID conversion; got {}".format(len(pd_df))
        )

    print("[R-only 4/4] Building 29x29 windows (all label 0)...", flush=True)
    as_strided = np.lib.stride_tricks.as_strided
    win = 29
    s = 29
    feature = as_strided(
        pd_df.canBits, ((len(pd_df) - win) // s + 1, win), (8 * s, 8)
    )
    df = pd.DataFrame(
        {
            "features": pd.Series(feature.tolist()),
            "label": np.zeros(len(feature), dtype=np.int64),
        },
        index=range(len(feature)),
    )

    print("[R-only] DONE — #Normal windows: {}".format(len(df)), flush=True)
    return df[["features", "label"]].reset_index().drop(["index"], axis=1)
