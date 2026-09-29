
import numpy as np

import pandas as pd

import torch

from torch.utils.data import Dataset



class SleepAwakeDataset(Dataset):

    def __init__(self, manifest_csv, split):

        self.df = pd.read_csv(manifest_csv)

        self.df = self.df[self.df["split"].astype(str).str.lower() == split.lower()].reset_index(drop=True)

        if len(self.df) == 0:

            raise RuntimeError(f"No rows found for split={split}")



    def __len__(self):

        return len(self.df)



    def __getitem__(self, idx):

        r = self.df.iloc[idx]

        z = np.load(r["cache_path"], allow_pickle=True)

        x = z["x"].astype(np.float32)

        mask = z["channel_mask"].astype(np.float32)

        y = np.int64(r["label"])



        # x shape: [channels, samples] = [128, 60*256]

        # convert to [channels, patches, samples_per_patch] = [128, 60, 256]

        fs = int(round(float(r["target_fs"])))

        context_sec = int(round(float(r["context_sec"])))

        x = x[:, :context_sec * fs]

        x = x.reshape(x.shape[0], context_sec, fs)



        return {

            "x": torch.from_numpy(x),

            "channel_mask": torch.from_numpy(mask),

            "label": torch.tensor(y, dtype=torch.long),

            "patient_id": str(r["patient_id"]),

            "state": str(r["state"]),

            "window_id": str(r["window_id"]),

        }

