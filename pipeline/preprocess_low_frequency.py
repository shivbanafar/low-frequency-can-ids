"""
Build TFRecords from the merged low-frequency CSV (low_frequency_can_dataset.csv)
using the same 29x29 windowing as preprocessing.py, with a single synthetic attack
name "MergedLowFreq" so train_test_split.py and train.py can run with --labels.

Typical flow:
  1) Generate CSV: low_frequency_dataset/generate_merged_low_frequency_dataset.py (or generate_low_frequency_dataset.py)
  2) python preprocess_low_frequency.py --csv ../low_frequency_dataset/low_frequency_can_dataset.csv --outdir ./Data/TFRecord
  3) python train_test_split.py --indir ./Data/TFRecord --outdir ./Data --attack_type MergedLowFreq --normal
     (output folder name includes ratios, e.g. Train_0.7_Labeled_0.15 with current train_test_split defaults)
  4) python train.py --model CAAE --data_dir ./Data/Train_0.7_Labeled_0.15/ --labels MergedLowFreq Normal --is_train
"""
import argparse
import json
import os
import sys

sys.stdout.reconfigure(line_buffering=True)

# Windowing uses preprocessing_core (no TensorFlow). TFRecords use tfrecord_utils (tfrecord package, no TF import).
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from preprocessing_core import preprocess, preprocess_r_only_windows, preprocess_hcrl_normal  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="TFRecords from merged low-frequency CAN CSV")
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Path to low_frequency_can_dataset.csv (default: ../low_frequency_dataset/low_frequency_can_dataset.csv)",
    )
    parser.add_argument(
        "--attack_name",
        type=str,
        default="MergedLowFreq",
        help="Synthetic attack name used in TFRecord filenames and train_test_split (default: MergedLowFreq)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="./Data/TFRecord",
        help="Output directory for TFRecord files and datainfo.txt (default: ./Data/TFRecord)",
    )
    parser.add_argument(
        "--normal_file",
        type=str,
        default=None,
        help=(
            "Path to HCRL normal_run_data.txt for normal windows. "
            "Recommended: follows the paper's method of using a separate attack-free capture for normal windows. "
            "Falls back to R-only extraction from --csv if not provided."
        ),
    )
    args = parser.parse_args()

    if args.csv is None:
        csv_path = os.path.normpath(os.path.join(_SCRIPT_DIR, "..", "low_frequency_dataset", "low_frequency_can_dataset.csv"))
    else:
        csv_path = os.path.abspath(args.csv)

    if not os.path.isfile(csv_path):
        print(f"CSV not found: {csv_path}", flush=True)
        sys.exit(1)

    # Same path style as preprocessing.py main() so datainfo keys match train_test_split.py lookups
    outdir = args.outdir.rstrip("/")
    os.makedirs(outdir, exist_ok=True)

    attack = args.attack_name
    print(f"Input: {csv_path}", flush=True)
    print(f"Output dir: {outdir} (attack name: {attack})", flush=True)

    # Attack TFRecords: windows where any frame is T (same as preprocess on full stream).
    df_full = preprocess(csv_path)
    df_attack = df_full[df_full["label"] == 1]

    # Normal TFRecords: use a separate attack-free capture (paper's method) when available.
    # This avoids the domain mismatch from extracting R-only rows out of a mixed R/T file.
    if args.normal_file:
        normal_file_path = os.path.abspath(args.normal_file)
        if not os.path.isfile(normal_file_path):
            print(f"Normal file not found: {normal_file_path}", flush=True)
            import sys; sys.exit(1)
        print(f"Normal source: {normal_file_path} (HCRL normal_run_data.txt)", flush=True)
        df_normal = preprocess_hcrl_normal(normal_file_path)
    else:
        print("Normal source: R-only windows from mixed CSV (fallback; --normal_file preferred)", flush=True)
        df_normal = preprocess_r_only_windows(csv_path)

    foutput_attack = "{}/{}".format(outdir, attack)
    foutput_normal = "{}/Normal_{}".format(outdir, attack)

    from tfrecord_utils import write_tfrecord  # noqa: E402

    print("Writing TFRecords (lightweight writer; no TensorFlow load for export)...", flush=True)
    write_tfrecord(df_attack, foutput_attack)
    write_tfrecord(df_normal, foutput_normal)

    data_info = {
        foutput_attack: int(df_attack.shape[0]),
        foutput_normal: int(df_normal.shape[0]),
    }
    datainfo_path = os.path.join(outdir, "datainfo.txt")
    with open(datainfo_path, "w") as f:
        json.dump(data_info, f)
    print(f"Wrote {datainfo_path}", flush=True)
    print("DONE:", data_info, flush=True)


if __name__ == "__main__":
    main()
