
import random



import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset





class SeizureWindowDataset(Dataset):

    def __init__(

        self,

        manifest_csv,

        split,

        context_sec=60,

        patch_sec=1,

        max_channels=128,

        standardize=True,

        seed=42,

    ):

        self.df = pd.read_csv(manifest_csv)

        self.df = self.df[self.df["split"] == split].reset_index(drop=True)



        if len(self.df) == 0:

            raise ValueError(f"No rows found for split={split}")



        self.context_sec = int(context_sec)

        self.patch_sec = int(patch_sec)

        self.seq_len = self.context_sec // self.patch_sec

        self.max_channels = int(max_channels)

        self.standardize = standardize

        self.seed = seed



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        rng = random.Random(self.seed + idx)



        data_path = row["data_path"]

        fs = int(round(float(row["fs"])))



        start_sample = int(round(float(row["start_sec"]) * fs))

        context_samples = int(fs * self.context_sec)

        patch_samples = int(fs * self.patch_sec)



        arr = np.load(data_path, mmap_mode="r")  # [channels, samples]

        c, t = arr.shape



        end_sample = min(start_sample + context_samples, t)

        x = np.asarray(arr[:, start_sample:end_sample], dtype=np.float32)



        if x.shape[1] < context_samples:

            pad = context_samples - x.shape[1]

            x = np.pad(x, ((0, 0), (0, pad)), mode="constant")



        if c > self.max_channels:

            ch_idx = sorted(rng.sample(range(c), self.max_channels))

            x = x[ch_idx]

            real_c = self.max_channels

        else:

            real_c = c



        if self.standardize:

            mean = x.mean(axis=1, keepdims=True)

            std = x.std(axis=1, keepdims=True) + 1e-6

            x = (x - mean) / std



        x = x.reshape(x.shape[0], self.seq_len, patch_samples)



        out = np.zeros((self.max_channels, self.seq_len, patch_samples), dtype=np.float32)

        channel_mask = np.zeros((self.max_channels,), dtype=np.float32)



        out[:real_c] = x[:real_c]

        channel_mask[:real_c] = 1.0



        return {

            "x": torch.from_numpy(out),

            "fs": torch.tensor(fs, dtype=torch.long),

            "channel_mask": torch.from_numpy(channel_mask),

            "label": torch.tensor(float(row["label"]), dtype=torch.float32),

            "run": str(row["run"]),

            "original_name": str(row["original_name"]),

            "start_sec": torch.tensor(float(row["start_sec"]), dtype=torch.float32),

        }

