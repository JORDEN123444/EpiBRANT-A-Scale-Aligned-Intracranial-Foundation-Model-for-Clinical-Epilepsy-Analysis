import random
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

import mne



class EpiBRANTPretrainDataset(Dataset):

    """
    BrainWave-style SEEG foundation model pretraining dataset.

    Input:
        global_pretrain_manifest.csv

    Output:
        signal:
            C x T

        fs:
            native sampling frequency

        metadata:
            dataset / subject / recording information
    """



    def __init__(
        self,
        manifest_path,
        window_seconds=60,
        min_channels=16
    ):

        self.df = pd.read_csv(
            manifest_path
        )


        self.window_seconds = window_seconds

        self.min_channels = min_channels



        # remove very short recordings

        self.df = self.df[
            self.df.duration_seconds >= window_seconds
        ].reset_index(drop=True)



        print("======================")
        print("EpiBRANT Dataset")
        print("======================")

        print(
            "Recordings:",
            len(self.df)
        )


        print(
            "Subjects:",
            self.df.subject_id.nunique()
        )



    def __len__(self):

        return len(self.df)



    def select_seeg_channels(
        self,
        raw,
        channel_names
    ):


        available = []


        for ch in channel_names:

            if ch in raw.ch_names:

                available.append(ch)



        return available



    def normalize(
        self,
        x
    ):

        """
        Channel-wise z-score

        x:
            C x T
        """

        mean = np.mean(
            x,
            axis=1,
            keepdims=True
        )


        std = np.std(
            x,
            axis=1,
            keepdims=True
        )


        x = (
            x - mean
        ) / (
            std + 1e-6
        )


        return x



    def __getitem__(
        self,
        index
    ):


        row = self.df.iloc[index]



        filepath = row["file"]



        fs = int(
            row["sampling_rate"]
        )



        # ------------------------------------------------
        # Load EDF
        # ------------------------------------------------


        raw = mne.io.read_raw_edf(

            filepath,

            preload=False,

            verbose=False,

            encoding="latin1"

        )



        # ------------------------------------------------
        # Select SEEG channels
        # ------------------------------------------------


        seeg_names = row[
            "seeg_channel_names"
        ].split(";")



        seeg_names = self.select_seeg_channels(
            raw,
            seeg_names
        )



        if len(seeg_names) < self.min_channels:

            raise RuntimeError(
                f"Too few SEEG channels: {filepath}"
            )



        raw.pick(
            seeg_names
        )



        total_samples = raw.n_times



        window_samples = (
            self.window_seconds * fs
        )



        # ------------------------------------------------
        # Random 60 sec crop
        # ------------------------------------------------


        start = random.randint(

            0,

            total_samples - window_samples

        )


        stop = (
            start + window_samples
        )



        signal = raw.get_data(

            start=start,

            stop=stop

        )



        # C x T

        signal = self.normalize(
            signal
        )



        signal = torch.tensor(

            signal,

            dtype=torch.float32

        )



        return {


            "signal":

                signal,


            "sampling_rate":

                fs,


            "dataset":

                row["dataset"],


            "subject_id":

                row["subject_id"],


            "session_id":

                row["session_id"],


            "recording_id":

                row["recording_id"]


        }