import argparse
from pathlib import Path
import pandas as pd
import mne

mne.set_log_level("WARNING")

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", required=True)
parser.add_argument("--out_csv", default="edf_channel_inspection.csv")
args = parser.parse_args()

data_dir = Path(args.data_dir)

rows = []

included_files = ["1-SEZ.edf", "2-SEZ.edf", "3-SEZ.edf"]

for fname in included_files:
    p = data_dir / fname

    print("\n" + "=" * 90)
    print("File:", p)

    if not p.exists():
        print("ERROR: File does not exist:", p)
        continue

    raw = mne.io.read_raw_edf(str(p), preload=False, verbose=False)

    fs = raw.info["sfreq"]
    dur = raw.n_times / fs

    print("Sampling rate:", fs)
    print("Duration sec:", dur)
    print("Number of channels:", len(raw.ch_names))
    print("EDF header measurement date:", raw.info.get("meas_date"))
    print("First 80 channels:")

    for ch in raw.ch_names[:80]:
        print(" ", ch)

    for idx, ch in enumerate(raw.ch_names):
        rows.append({
            "edf_file": fname,
            "channel_index": idx,
            "channel_name": ch,
            "sampling_rate": fs,
            "duration_sec": dur,
            "n_channels": len(raw.ch_names),
            "meas_date": str(raw.info.get("meas_date")),
        })

out_csv = data_dir / args.out_csv
pd.DataFrame(rows).to_csv(out_csv, index=False)

print("\nNOTE: 4-SEZ.edf is intentionally excluded.")
print("Saved full channel inspection CSV:", out_csv)