
import argparse

import re

from pathlib import Path

import pandas as pd

import yaml



def safe_name(x):

    x = str(x)

    x = re.sub(r"[^A-Za-z0-9_\-]+", "_", x)

    return x.strip("_")



def has_both_labels(df):

    return set(df["label"].astype(int).unique()) == {0, 1}



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--base_config", required=True)

    ap.add_argument("--manifest", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--require_both_labels", action="store_true")

    args = ap.parse_args()



    base_config = yaml.safe_load(open(args.base_config))

    manifest = pd.read_csv(args.manifest)



    out_dir = Path(args.out_dir)

    manifest_dir = out_dir / "manifests"

    config_dir = out_dir / "configs"

    run_root = out_dir / "runs"



    manifest_dir.mkdir(parents=True, exist_ok=True)

    config_dir.mkdir(parents=True, exist_ok=True)

    run_root.mkdir(parents=True, exist_ok=True)



    patients = sorted(manifest["patient_id"].astype(str).unique())



    print("\nPatients found:")

    for p in patients:

        sub = manifest[manifest["patient_id"].astype(str) == p]

        print(f"  {p}: rows={len(sub)}, label_counts={sub['label'].value_counts().sort_index().to_dict()}")



    made = []



    for p in patients:

        val_df = manifest[manifest["patient_id"].astype(str) == p]

        train_df = manifest[manifest["patient_id"].astype(str) != p]



        if len(val_df) == 0 or len(train_df) == 0:

            print(f"\nSKIP {p}: empty train or validation.")

            continue



        if not has_both_labels(train_df):

            print(f"\nSKIP {p}: training set does not contain both awake and sleep.")

            continue



        if args.require_both_labels and not has_both_labels(val_df):

            print(f"\nSKIP {p}: validation patient does not contain both awake and sleep.")

            continue



        fold_name = f"lopo_val_{safe_name(p)}"



        fold_manifest = manifest.copy()

        fold_manifest["split"] = "train"

        fold_manifest.loc[fold_manifest["patient_id"].astype(str) == p, "split"] = "val"



        fold_manifest_path = manifest_dir / f"{fold_name}.csv"

        fold_manifest.to_csv(fold_manifest_path, index=False)



        cfg = dict(base_config)

        cfg["data"] = dict(base_config["data"])

        cfg["train"] = dict(base_config["train"])

        cfg["data"]["manifest_csv"] = str(fold_manifest_path)

        cfg["train"]["out_dir"] = str(run_root / fold_name)



        cfg_path = config_dir / f"{fold_name}.yaml"

        with open(cfg_path, "w") as f:

            yaml.safe_dump(cfg, f, sort_keys=False)



        made.append({

            "fold": fold_name,

            "validation_patient": p,

            "config_path": str(cfg_path),

            "manifest_path": str(fold_manifest_path),

            "train_rows": len(train_df),

            "val_rows": len(val_df),

            "train_awake": int((train_df["label"].astype(int) == 0).sum()),

            "train_sleep": int((train_df["label"].astype(int) == 1).sum()),

            "val_awake": int((val_df["label"].astype(int) == 0).sum()),

            "val_sleep": int((val_df["label"].astype(int) == 1).sum()),

        })



        print("\nCreated:", fold_name)

        print("  validation patient:", p)

        print("  train rows:", len(train_df), "| val rows:", len(val_df))

        print("  train label counts:", train_df["label"].value_counts().sort_index().to_dict())

        print("  val label counts:", val_df["label"].value_counts().sort_index().to_dict())



    fold_table = pd.DataFrame(made)

    fold_table_path = out_dir / "lopo_fold_table.csv"

    fold_table.to_csv(fold_table_path, index=False)



    print("\nSaved fold table:")

    print(fold_table_path)

    print("\nTotal folds created:", len(made))



    if len(made) == 0:

        raise RuntimeError("No LOPO folds created. Check patient labels in the cached manifest.")



if __name__ == "__main__":

    main()

