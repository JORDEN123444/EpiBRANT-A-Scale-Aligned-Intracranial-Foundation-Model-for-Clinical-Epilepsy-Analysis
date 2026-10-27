
import argparse

import json

from pathlib import Path



import mne

import numpy as np





BAD_KEYWORDS = [

    "DC", "SPO2", "POL SPO2", "EKG", "ECG", "BP", "TRIG", "MARK", "EVENT",

    "EMG", "EOG", "RESP", "PLETH", "PULSE", "TEMP", "PHOTO"

]





def is_bad_channel(ch):

    u = ch.upper().replace("-", " ").replace("_", " ")

    return any(k in u for k in BAD_KEYWORDS)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--raw_dir", required=True)

    parser.add_argument("--out_root", required=True)

    parser.add_argument("--l_freq", type=float, default=0.5)

    parser.add_argument("--h_freq", type=float, default=128.0)

    parser.add_argument("--notch", type=float, default=50.0)

    args = parser.parse_args()



    raw_dir = Path(args.raw_dir)

    out_root = Path(args.out_root)

    out_root.mkdir(parents=True, exist_ok=True)



    edf_files = sorted(raw_dir.glob("*.edf"))

    print("Found EDF files:", len(edf_files))



    if len(edf_files) == 0:

        raise RuntimeError(f"No EDF files found in {raw_dir}")



    for edf_path in edf_files:

        run_name = edf_path.stem

        out_dir = out_root / f"run_{run_name}"

        out_dir.mkdir(parents=True, exist_ok=True)



        print("\nReading:", edf_path)



        raw = mne.io.read_raw_edf(

            str(edf_path),

            preload=True,

            verbose=False,

            encoding="latin1",

        )



        fs = float(raw.info["sfreq"])

        ch_names = list(raw.ch_names)



        keep_idx = []

        keep_ch = []



        for i, ch in enumerate(ch_names):

            if not is_bad_channel(ch):

                keep_idx.append(i)

                keep_ch.append(ch)



        if len(keep_idx) == 0:

            print("[SKIP] no valid SEEG channels:", edf_path)

            continue



        raw.pick(keep_idx)



        # Notch filter

        try:

            freqs = [args.notch]

            if fs / 2.0 > args.notch * 2:

                freqs.append(args.notch * 2)

            raw.notch_filter(freqs=freqs, verbose=False)

        except Exception as e:

            print("[WARN] notch failed:", e)



        # Bandpass filter

        h_freq = min(args.h_freq, fs / 2.0 - 1.0)

        try:

            raw.filter(l_freq=args.l_freq, h_freq=h_freq, verbose=False)

        except Exception as e:

            print("[WARN] bandpass failed:", e)



        data = raw.get_data().astype("float32")



        # Convert to microvolts if data are in volts

        if np.nanmedian(np.abs(data)) < 1e-3:

            data = data * 1e6



        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)



        np.save(out_dir / "data.npy", data)



        with open(out_dir / "channels.txt", "w") as f:

            for ch in raw.ch_names:

                f.write(str(ch) + "\n")



        info = {

            "source_edf": str(edf_path),

            "run": run_name,

            "sampling_rate": fs,

            "n_channels": int(data.shape[0]),

            "n_samples": int(data.shape[1]),

            "duration_sec": float(data.shape[1] / fs),

            "l_freq": args.l_freq,

            "h_freq": h_freq,

            "notch": args.notch,

        }



        with open(out_dir / "info.json", "w") as f:

            json.dump(info, f, indent=2)



        print("Saved:", out_dir)

        print("shape:", data.shape, "fs:", fs, "duration_sec:", info["duration_sec"])





if __name__ == "__main__":

    main()

