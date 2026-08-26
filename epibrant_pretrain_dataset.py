import torch
from torch.utils.data import Dataset

import pandas as pd
import numpy as np
import mne
import random



class EpiBRANTPretrainDataset(Dataset):

    """
    Dataset loader for EpiBRANT self-supervised pretraining.


    Input:

        global manifest.csv


    Output:

        signal:
            C x T

        mask:
            temporal-channel mask

        metadata

    """



    def __init__(
        self,
        manifest,
        max_channels=256,
        window_seconds=60
    ):


        self.df = pd.read_csv(
            manifest
        )


        self.max_channels=max_channels

        self.window_seconds=window_seconds



        self.bad_keywords=[

            "ECG",
            "EKG",
            "EMG",
            "EOG",
            "RESP",
            "SPO2",
            "STATUS"

        ]



    def __len__(self):

        return len(self.df)



    def select_seeg_channels(
        self,
        raw
    ):


        keep=[]


        for ch in raw.ch_names:


            flag=False


            for bad in self.bad_keywords:

                if bad in ch.upper():

                    flag=True


            if not flag:

                keep.append(ch)



        raw.pick(
            keep
        )


        return raw



    def normalize(
        self,
        x
    ):


        mean=x.mean(
            axis=1,
            keepdims=True
        )


        std=x.std(
            axis=1,
            keepdims=True
        )


        x=(x-mean)/(std+1e-6)


        return x



    def pad_channels(
        self,
        x
    ):


        C,T=x.shape



        if C > self.max_channels:

            x=x[:self.max_channels]


        elif C < self.max_channels:


            pad=np.zeros(

                (
                    self.max_channels-C,
                    T

                ),

                dtype=np.float32

            )


            x=np.concatenate(

                [
                    x,
                    pad
                ],

                axis=0

            )


        return x



    def __getitem__(
        self,
        idx
    ):


        row=self.df.iloc[idx]



        path=row.file



        raw=mne.io.read_raw_edf(

            path,

            preload=True,

            verbose=False

        )



        fs=int(
            raw.info["sfreq"]
        )



        raw=self.select_seeg_channels(
            raw
        )



        total_samples=raw.n_times



        window_samples=int(

            fs*self.window_seconds

        )



        if total_samples > window_samples:


            start=random.randint(

                0,

                total_samples-window_samples

            )


        else:

            start=0



        x=raw.get_data(

            start=start,

            stop=start+window_samples

        )



        # normalize

        x=self.normalize(
            x
        )


        # channel padding

        x=self.pad_channels(
            x
        )



        x=torch.tensor(

            x,

            dtype=torch.float32

        )



        return {

            "signal":x,

            "fs":fs,

            "subject":row.subject_id

        }