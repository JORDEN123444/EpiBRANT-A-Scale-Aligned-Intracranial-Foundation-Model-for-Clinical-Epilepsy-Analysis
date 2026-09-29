
import argparse

from pathlib import Path

import hashlib

import numpy as np

import pandas as pd

import mne

from scipy.signal import butter, sosfiltfilt, iirnotch, filtfilt, resample_poly

from math import gcd



mne.set_log_level("WARNING")



def bandpass(data, fs, low=0.5, high=120.0):

    nyq = fs / 2.0

    high = min(high, nyq - 1.0)

    if high <= low:

        return data

    sos = butter(4, [low / nyq, high / nyq], btype="bandpass", output="sos")

    return sosfiltfilt(sos, data, axis=-1)



def notch(data, fs, freq=50.0, q=30.0):

    nyq = fs / 2.0

    for f in [freq, freq * 2]:

        if f < nyq:

            b, a = iirnotch(w0=f / nyq, Q=q)

            data = filtfilt(b, a, data, axis=-1)

    return data



def resample_block(data, fs_in, fs_out):

    if abs(fs_in - fs_out) < 1e-6:

        return data.astype(np.float32)

    fs_in_i = int(round(fs_in))

    fs_out_i = int(round(fs_out))

    g = gcd(fs_in_i, fs_out_i)

    up = fs_out_i // g

    down = fs_in_i // g

    return resample_poly(data, up, down, axis=-1).astype(np.float32)



def robust_normalize(data):

    med = np.median(data, axis=-1, keepdims=True)

    q25 = np.percentile(data, 25, axis=-1, keepdims=True)

    q75 = np.percentile(data, 75, axis=-1, keepdims=True)

    iqr = q75 - q25

    scale = iqr / 1.349

    scale[scale < 1e-6] = 1.0

    data = (data - med) / scale

    data = np.clip(data, -10.0, 10.0)

    return data.astype(np.float32)



def pad_or_crop_channels(data, channels, max_channels):

    c, t = data.shape

    if c >= max_channels:

        return data[:max_channels], channels[:max_channels], np.ones(max_channels, dtype=np.float32)

    out = np.zeros((max_channels, t), dtype=np.float32)

    mask = np.zeros(max_channels, dtype=np.float32)

    out[:c] = data

    mask[:c] = 1.0

    return out, channels, mask



def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--manifest", required=True)

    ap.add_argument("--out_dir", required=True)

    ap.add_argument("--target_fs", type=float, default=256.0)

    ap.add_argument("--max_channels", type=int, default=128)

    ap.add_argument("--bandpass_low", type=float, default=0.5)

    ap.add_argument("--bandpass_high", type=float, default=120.0)

    ap.add_argument("--notch_freq", type=float, default=50.0)

    args = ap.parse_args()



    out_dir = Path(args.out_dir)

    cache_dir = out_dir / "cache_preprocessed_60s_multichannel"

    cache_dir.mkdir(parents=True, exist_ok=True)



    df = pd.read_csv(args.manifest).reset_index(drop=True)

    cache_paths = []



    print("Manifest rows:", len(df))

    print("Cache dir:", cache_dir)

    print("Band-pass:", args.bandpass_low, "-", args.bandpass_high, "Hz")

    print("Notch:", args.notch_freq, "Hz plus harmonic if valid")

    print("Final target fs for pretrained model:", args.target_fs)

    print("Max channels:", args.max_channels)



    raw_cache = {}



    for i, r in df.iterrows():

        edf_path = str(r["edf_path"])

        start_sec = float(r["segment_start_sec"])

        end_sec = float(r["segment_end_sec"])

        raw_channels = str(r["raw_channels"]).split("|")

        clean_channels = str(r["clean_channels"]).split("|")



        key = f"{edf_path}|{start_sec:.4f}|{end_sec:.4f}|{args.target_fs}|{args.max_channels}|{args.bandpass_low}-{args.bandpass_high}|notch{args.notch_freq}"

        h = hashlib.md5(key.encode()).hexdigest()

        cache_path = cache_dir / f"{h}.npz"



        if cache_path.exists():

            cache_paths.append(str(cache_path))

            continue



        if edf_path not in raw_cache:

            try:

                raw_cache[edf_path] = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

            except Exception as e:

                print("Standard EDF read failed. Retrying with encoding='latin1'")

                print("EDF:", edf_path)

                print("Reason:", repr(e))

                raw_cache[edf_path] = mne.io.read_raw_edf(edf_path, preload=False, verbose=False, encoding="latin1")



        raw = raw_cache[edf_path]

        fs = float(raw.info["sfreq"])

        start = int(round(start_sec * fs))

        stop = int(round(end_sec * fs))



        data = raw.get_data(picks=raw_channels, start=start, stop=stop)

        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)



        data = bandpass(data, fs, args.bandpass_low, args.bandpass_high)

        data = notch(data, fs, args.notch_freq)

        data = resample_block(data, fs, args.target_fs)



        expected_len = int(round((end_sec - start_sec) * args.target_fs))

        if data.shape[-1] > expected_len:

            data = data[:, :expected_len]

        elif data.shape[-1] < expected_len:

            pad = expected_len - data.shape[-1]

            data = np.pad(data, ((0, 0), (0, pad)), mode="constant")



        data = robust_normalize(data)

        data, used_channels, channel_mask = pad_or_crop_channels(data, clean_channels, args.max_channels)



        np.savez_compressed(

            cache_path,

            x=data.astype(np.float32),

            channel_mask=channel_mask.astype(np.float32),

            channels=np.array(used_channels, dtype=object),

            label=np.array(int(r["label"]), dtype=np.int64),

            patient_id=str(r["patient_id"]),

            state=str(r["state"]),

            fs=np.array(args.target_fs, dtype=np.float32),

        )



        cache_paths.append(str(cache_path))



        if (i + 1) % 20 == 0:

            print(f"Cached {i+1}/{len(df)}")



    df["cache_path"] = cache_paths

    df["target_fs"] = args.target_fs

    df["bandpass_low_hz"] = args.bandpass_low

    df["bandpass_high_hz"] = args.bandpass_high

    df["notch_hz"] = args.notch_freq

    df["normalization"] = "per_channel_robust_median_iqr_clip_10"

    df["max_channels"] = args.max_channels



    out_manifest = out_dir / "sleep_awake_manifest_cached.csv"

    df.to_csv(out_manifest, index=False)



    print("\nSaved cached manifest:")

    print(out_manifest)

    print("\nSplit x label:")

    print(pd.crosstab(df["split"], df["label"]).to_string())



if __name__ == "__main__":

    main()

