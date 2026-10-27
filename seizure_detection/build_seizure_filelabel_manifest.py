
import argparse

import json

import re

from pathlib import Path



import numpy as np

import pandas as pd





def get_index(original_name):

    m = re.search(r"_(\d+)$", original_name)

    if m:

        return int(m.group(1))

    return 999





def assign_split(original_name):

    idx = get_index(original_name)



    if idx in [1, 2, 3]:

        return "train"

    elif idx == 4:

        return "val"

    elif idx == 5:

        return "test"

    else:

        return "train"





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--processed_root", required=True)

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--window_sec", type=float, default=60.0)

    parser.add_argument("--stride_sec", type=float, default=30.0)

    parser.add_argument("--max_windows_per_file", type=int, default=200)

    args = parser.parse_args()



    root = Path(args.processed_root)

    rows = []



    run_dirs = sorted(root.glob("run_*"))

    print("Found processed runs:", len(run_dirs))



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



        original_name = info["original_name"]

        label = int(info["label"])

        split = assign_split(original_name)



        starts = []

        t = 0.0



        while t + args.window_sec <= duration_sec:

            starts.append(t)

            t += args.stride_sec



        if len(starts) == 0 and duration_sec > 5:

            starts = [0.0]



        if len(starts) > args.max_windows_per_file:

            idx = np.linspace(0, len(starts) - 1, args.max_windows_per_file).astype(int)

            starts = [starts[i] for i in idx]



        for start_sec in starts:

            rows.append({

                "dataset": "seizure_detection_filelabel",

                "patient": info.get("patient", "Patient_1"),

                "subject": info.get("patient", "Patient_1"),

                "run": info["run"],

                "original_name": original_name,

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

        raise RuntimeError("No windows created. Check processed_root and processed data folders.")



    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    df.to_csv(args.out_csv, index=False)



    print("\nSaved:", args.out_csv)

    print("Total windows:", len(df))



    print("\nSplit/label counts:")

    print(df.groupby(["split", "label"]).size())



    print("\nRuns:")

    print(df.groupby(["split", "original_name", "label"]).size())





if __name__ == "__main__":

    main()

