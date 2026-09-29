
import argparse

from pathlib import Path

import mne



mne.set_log_level("WARNING")



parser = argparse.ArgumentParser()

parser.add_argument("--data_dir", required=True)

args = parser.parse_args()



data_dir = Path(args.data_dir)



for fname in ["1-SEZ.edf", "2-SEZ.edf", "3-SEZ.edf"]:

    p = data_dir / fname

    print("\n" + "="*90)

    print("File:", p)

    raw = mne.io.read_raw_edf(str(p), preload=False, verbose=False)

    fs = raw.info["sfreq"]

    dur = raw.n_times / fs

    print("Sampling rate:", fs)

    print("Duration sec:", dur)

    print("Number of channels:", len(raw.ch_names))

    print("First 80 channels:")

    for ch in raw.ch_names[:80]:

        print(" ", ch)



print("\nNOTE: 4-SEZ.edf is intentionally excluded.")

