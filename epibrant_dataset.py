import torch
import numpy as np
import mne
from torch.utils.data import Dataset


class EpiBRANTDataset(Dataset):


    def __init__(
        self,
        manifest,
        window_seconds=60,
        max_channels=288
    ):

        self.manifest = manifest

        self.window_seconds = window_seconds

        self.max_channels=max_channels



    def __len__(self):

        return len(self.manifest)



    def __getitem__(self,index):


        row=self.manifest.iloc[index]


        raw=mne.io.read_raw_edf(
            row.file,
            preload=True,
            verbose=False
        )


        fs=int(
            raw.info["sfreq"]
        )


        data=raw.get_data()


        # SEEG channels x samples

        samples=window_samples = (
            fs*self.window_seconds
        )



        # random 60 second window

        if data.shape[1] > window_samples:

            start=np.random.randint(
                0,
                data.shape[1]-window_samples
            )

            data=data[
                :,
                start:start+window_samples
            ]

        else:

            pad=(
                window_samples-data.shape[1]
            )

            data=np.pad(
                data,
                ((0,0),(0,pad))
            )



        channels=data.shape[0]



        # channel padding

        x=np.zeros(
            (
            self.max_channels,
            window_samples
            ),
            dtype=np.float32
        )


        mask=np.zeros(
            self.max_channels,
            dtype=np.float32
        )



        use=min(
            channels,
            self.max_channels
        )


        x[:use]=data[:use]

        mask[:use]=1



        return {

            "signal":
                torch.tensor(x),

            "channel_mask":
                torch.tensor(mask),

            "sampling_rate":
                fs

        }