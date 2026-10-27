
import argparse

import json

import re

from pathlib import Path



import numpy as np

import pandas as pd





def natural_key(x):

    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(x))]





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--processed_root", required=True)

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--window_sec", type=float, default=5.0)

    parser.add_argument("--stride_sec", type=float, default=2.0)

    parser.add_argument("--max_windows_per_file", type=int, default=1000)

    parser.add_argument("--train_patients", nargs="+", required=True)

    parser.add_argument("--val_patients", nargs="+", required=True)

    parser.add_argument("--test_patients", nargs="+", required=True)

    parser.add_argument("--balance_splits", action="store_true")

    args = parser.parse_args()



    root = Path(args.processed_root)



    if not root.exists():

        raise FileNotFoundError(f"Processed root not found: {root}")



    train_patients = set(args.train_patients)

    val_patients = set(args.val_patients)

    test_patients = set(args.test_patients)



    overlap = (train_patients & val_patients) | (train_patients & test_patients) | (val_patients & test_patients)

    if overlap:

        raise ValueError(f"Patients appear in more than one split: {overlap}")



    patient_dirs = sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: natural_key(p.name))

    found_patients = [p.name for p in patient_dirs]



    print("Patients found:", found_patients)

    print("Train patients:", sorted(train_patients, key=natural_key))

    print("Val patients:", sorted(val_patients, key=natural_key))

    print("Test patients:", sorted(test_patients, key=natural_key))



    rows = []



    for patient_dir in patient_dirs:

        patient = patient_dir.name



        if patient in train_patients:

            split = "train"

        elif patient in val_patients:

            split = "val"

        elif patient in test_patients:

            split = "test"

        else:

            continue



        run_dirs = sorted(patient_dir.glob("run_*"), key=lambda p: natural_key(p.name))

        print(f"\n{patient} | split={split} | runs={len(run_dirs)}")



        for run_dir in run_dirs:

            data_path = run_dir / "data.npy"

            info_path = run_dir / "info.json"



            if not data_path.exists() or not info_path.exists():

                print("[SKIP missing data/info]", run_dir)

                continue



            with open(info_path, "r") as f:

                info = json.load(f)



            arr = np.load(data_path, mmap_mode="r")

            fs = float(info["sampling_rate"])

            duration_sec = arr.shape[1] / fs

            label = int(info["label"])



            starts = []

            t = 0.0



            while t + args.window_sec <= duration_sec:

                starts.append(t)

                t += args.stride_sec



            if len(starts) == 0 and duration_sec > 2:

                starts = [0.0]



            if len(starts) > args.max_windows_per_file:

                idx = np.linspace(0, len(starts) - 1, args.max_windows_per_file).astype(int)

                starts = [starts[i] for i in idx]



            for start_sec in starts:

                rows.append({

                    "dataset": "patientwise_seizure_detection_custom_split",

                    "patient": patient,

                    "subject": patient,

                    "run": info["run"],

                    "original_name": info["original_name"],

                    "data_path": str(data_path),

                    "fs": fs,

                    "channels": int(arr.shape[0]),

                    "samples": int(arr.shape[1]),

                    "duration_sec": float(duration_sec),

                    "start_sec": float(start_sec),

                    "end_sec": float(start_sec + args.window_sec),

                    "label": label,

                    "split": split,

                })



    df = pd.DataFrame(rows)



    if len(df) == 0:

        raise RuntimeError("No windows were created. Check processed_root and patient names.")



    if args.balance_splits:

        balanced_parts = []



        for split in ["train", "val", "test"]:

            dsplit = df[df["split"] == split].copy()

            counts = dsplit["label"].value_counts().to_dict()



            print(f"\nBefore balancing {split}:", counts)



            if 0 not in counts or 1 not in counts:

                print(f"[WARNING] {split} does not contain both classes. Keeping unbalanced.")

                balanced_parts.append(dsplit)

                continue



            n = min(counts[0], counts[1])



            for label in [0, 1]:

                part = dsplit[dsplit["label"] == label].sample(n=n, random_state=42)

                balanced_parts.append(part)



        df = pd.concat(balanced_parts).sample(frac=1.0, random_state=42).reset_index(drop=True)



    out_csv = Path(args.out_csv)

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(out_csv, index=False)



    print("\nSaved manifest:", out_csv)

    print("Total windows:", len(df))



    print("\nFinal split/label counts:")

    print(df.groupby(["split", "label"]).size())



    print("\nPatient contribution:")

    print(df.groupby(["split", "patient", "label"]).size())



    print("\nFile contribution:")

    print(df.groupby(["split", "patient", "original_name", "label"]).size())





if __name__ == "__main__":

    main()

