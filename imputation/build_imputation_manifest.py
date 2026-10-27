
import argparse

import json

from pathlib import Path



import numpy as np

import pandas as pd





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--processed_root", required=True)

    parser.add_argument("--out_csv", required=True)

    args = parser.parse_args()



    root = Path(args.processed_root)



    split_map = {

        "run_1": "train",

        "run_2": "train",

        "run_3": "train",

        "run_4": "train",

        "run_5": "train",

        "run_7": "val",

        "run_8": "test",

    }



    rows = []



    for run_dir in sorted(root.glob("run_*")):

        data_path = run_dir / "data.npy"

        info_path = run_dir / "info.json"



        if not data_path.exists() or not info_path.exists():

            print("[SKIP]", run_dir)

            continue



        with open(info_path, "r") as f:

            info = json.load(f)



        run = run_dir.name

        split = split_map.get(run, "train")



        arr = np.load(data_path, mmap_mode="r")

        fs = float(info["sampling_rate"])

        duration_sec = arr.shape[1] / fs



        rows.append({

            "dataset": "imputation_rawseeg",

            "subject": "imputation_subject_001",

            "run": run,

            "data_path": str(data_path),

            "run_dir": str(run_dir),

            "fs": fs,

            "channels": int(arr.shape[0]),

            "samples": int(arr.shape[1]),

            "duration_sec": float(duration_sec),

            "split": split,

        })



    df = pd.DataFrame(rows)

    df.to_csv(args.out_csv, index=False)



    print("Saved:", args.out_csv)

    print("\nManifest:")

    print(df)



    print("\nDuration hours by split:")

    print(df.groupby("split")["duration_sec"].sum() / 3600)



    print("\nNumber of runs by split:")

    print(df["split"].value_counts())





if __name__ == "__main__":

    main()

