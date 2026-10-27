
import random



import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset





class ImputationWindowDataset(Dataset):

    def __init__(

        self,

        manifest_csv,

        split,

        context_sec=60,

        patch_sec=1,

        max_channels=128,

        windows_per_run=200,

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

        self.windows_per_run = int(windows_per_run)

        self.standardize = standardize

        self.seed = seed

        self.split = split



        self.index = []

        for row_id in range(len(self.df)):

            for k in range(self.windows_per_run):

                self.index.append((row_id, k))



    def __len__(self):

        return len(self.index)



    def __getitem__(self, idx):

        row_id, k = self.index[idx]

        row = self.df.iloc[row_id]



        rng = random.Random(self.seed + idx)



        data_path = row["data_path"]

        fs = int(round(float(row["fs"])))



        context_samples = int(fs * self.context_sec)

        patch_samples = int(fs * self.patch_sec)



        arr = np.load(data_path, mmap_mode="r")  # [C, T]

        c, t = arr.shape



        if t <= context_samples:

            start = 0

        else:

            start = rng.randint(0, t - context_samples)



        x = np.asarray(arr[:, start:start + context_samples], dtype=np.float32)



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

            "run": str(row["run"]),

            "start_sample": torch.tensor(start, dtype=torch.long),

        }

