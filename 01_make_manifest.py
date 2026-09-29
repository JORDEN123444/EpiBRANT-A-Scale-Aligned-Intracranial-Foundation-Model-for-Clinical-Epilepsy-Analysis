
import argparse

import re

from pathlib import Path

import numpy as np

import pandas as pd

import mne



mne.set_log_level("WARNING")



NON_SEEG_WORDS = [

    "ECG", "EKG", "EMG", "EOG", "SPO2", "SAO2", "PLETH", "RESP",

    "TRIG", "STIM", "EVENT", "MARK", "ANNOT", "DC", "GND", "REF"

]



def clean_channel_name(ch):

    s = str(ch).strip().upper()

    s = re.sub(r"^(EEG|POL)\s+", "", s)

    s = s.replace(" ", "")

    s = s.replace("–", "-")

    s = re.sub(r"-REF$", "", s)

    return s



def is_seeg_channel(ch):

    s = clean_channel_name(ch)

    if any(w in s for w in NON_SEEG_WORDS):

        return False

    return re.match(r"^[A-Z]+\'?\d+$", s) is not None



def edf_files(folder):

    files = []

    for p in sorted(folder.iterdir()):

        if p.is_file() and (p.name.lower().endswith(".edf") or p.name.lower().endswith("_edf")):

            files.append(p)

    return files



def make_windows(duration, window_sec, stride_sec, max_windows):

    if duration < window_sec:

        return []

    starts = np.arange(0, duration - window_sec + 1e-6, stride_sec).astype(float).tolist()

    if max_windows > 0 and len(starts) > max_windows:

        idx = np.linspace(0, len(starts) - 1, max_windows).round().astype(int)

        starts = [starts[i] for i in idx]

    return starts



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--root", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--window_sec", type=float, default=60.0)

    ap.add_argument("--stride_sec", type=float, default=30.0)

    ap.add_argument("--max_windows_per_file", type=int, default=80)

    ap.add_argument("--val_patients", default="patient9")

    args = ap.parse_args()



    root = Path(args.root)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    val_patients = set(x.strip() for x in args.val_patients.split(",") if x.strip())



    rows = []

    rec_rows = []

    ch_rows = []



    patient_dirs = [p for p in sorted(root.iterdir()) if p.is_dir() and p.name.lower().startswith(("patient", "pateint"))]

    if not patient_dirs:

        raise RuntimeError(f"No patient folders found in {root}")



    for patient_dir in patient_dirs:

        patient_id = patient_dir.name

        split = "val" if patient_id in val_patients else "train"



        for state, label in [("awake", 0), ("sleep", 1)]:

            state_dir = patient_dir / state

            if not state_dir.exists():

                print(f"WARNING: missing folder: {state_dir}")

                continue



            for edf in edf_files(state_dir):

                print("\n" + "=" * 100)

                print("Patient:", patient_id, "| State:", state, "| Label:", label, "| Split:", split)

                print("EDF:", edf)



                try:

                    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False)

                except Exception as e:

                    print("Standard EDF read failed. Retrying with encoding='latin1'")

                    print("Reason:", repr(e))

                    raw = mne.io.read_raw_edf(str(edf), preload=False, verbose=False, encoding="latin1")

                fs = float(raw.info["sfreq"])

                duration = raw.n_times / fs



                kept_raw = []

                kept_clean = []

                for ch in raw.ch_names:

                    clean = clean_channel_name(ch)

                    keep = is_seeg_channel(ch)

                    ch_rows.append({

                        "patient_id": patient_id,

                        "state": state,

                        "label": label,

                        "edf_path": str(edf),

                        "raw_channel_name": ch,

                        "clean_channel_name": clean,

                        "status": "kept" if keep else "dropped",

                    })

                    if keep:

                        kept_raw.append(ch)

                        kept_clean.append(clean)



                starts = make_windows(duration, args.window_sec, args.stride_sec, args.max_windows_per_file)



                print("Sampling rate:", fs)

                print("Duration sec:", round(duration, 2))

                print("Total channels:", len(raw.ch_names))

                print("SEEG channels kept:", len(kept_raw))

                print("60-sec windows:", len(starts))



                rec_rows.append({

                    "patient_id": patient_id,

                    "state": state,

                    "label": label,

                    "split": split,

                    "edf_path": str(edf),

                    "raw_file": edf.name,

                    "fs_original": fs,

                    "duration_sec": duration,

                    "n_total_channels": len(raw.ch_names),

                    "n_seeg_channels": len(kept_raw),

                    "n_windows": len(starts),

                })



                if len(kept_raw) == 0:

                    print("WARNING: no SEEG channels kept, skipping EDF.")

                    continue



                for wi, start_sec in enumerate(starts):

                    rows.append({

                        "patient_id": patient_id,

                        "state": state,

                        "label": label,

                        "split": split,

                        "edf_path": str(edf),

                        "raw_file": edf.name,

                        "window_id": f"{patient_id}_{state}_{edf.stem}_w{wi:04d}",

                        "segment_start_sec": float(start_sec),

                        "segment_end_sec": float(start_sec + args.window_sec),

                        "context_sec": float(args.window_sec),

                        "raw_channels": "|".join(kept_raw),

                        "clean_channels": "|".join(kept_clean),

                        "n_channels": len(kept_raw),

                    })



    manifest = pd.DataFrame(rows)

    rec = pd.DataFrame(rec_rows)

    ch_report = pd.DataFrame(ch_rows)



    if manifest.empty:

        raise RuntimeError("Manifest is empty. Check folders and EDF files.")



    manifest_path = out_dir / "sleep_awake_manifest.csv"

    rec_path = out_dir / "recording_summary.csv"

    ch_path = out_dir / "channel_filter_report.csv"



    manifest.to_csv(manifest_path, index=False)

    rec.to_csv(rec_path, index=False)

    ch_report.to_csv(ch_path, index=False)



    print("\nSaved:")

    print(manifest_path)

    print(rec_path)

    print(ch_path)



    print("\nSplit x label:")

    print(pd.crosstab(manifest["split"], manifest["label"]).to_string())



    print("\nPatient x state:")

    print(manifest.groupby(["patient_id", "state"]).size().to_string())



if __name__ == "__main__":

    main()

