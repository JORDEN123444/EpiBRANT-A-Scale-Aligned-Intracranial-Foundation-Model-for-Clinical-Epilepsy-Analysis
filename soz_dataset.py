from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class SOZLocalizationDataset(Dataset):
    """
    EpiBRANT downstream dataset for clinical SOZ localization.

    Each sample:
        One SEEG recording segment/patient.

    Input:
        signal:
            [288, T]

        channel_mask:
            [288]

        fs:
            native sampling frequency


    Output:
        soz_label:
            [288]

            1 = SOZ contact
            0 = non-SOZ contact


    Data source:
        Omni-iEEG BIDS dataset

    Required manifest columns:

        patient_id
        file
        channels_file
        start_sample
        stop_sample

    """


    def __init__(
        self,
        manifest_path,
        max_channels=288,
    ):

        super().__init__()

        self.manifest_path = Path(
            manifest_path
        )

        self.max_channels = int(
            max_channels
        )


        self.df = pd.read_csv(
            self.manifest_path
        )


        required = [

            "patient_id",
            "file",
            "channels_file",
            "start_sample",
            "stop_sample",

        ]


        for col in required:

            if col not in self.df.columns:

                raise RuntimeError(
                    f"Missing manifest column: {col}"
                )


        if len(self.df) == 0:

            raise RuntimeError(
                "SOZ manifest is empty"
            )



    # =====================================================
    # LENGTH
    # =====================================================

    def __len__(self):

        return len(self.df)



    # =====================================================
    # LOAD CHANNEL TSV
    # =====================================================

    def load_channel_info(
        self,
        channels_file
    ):


        channels = pd.read_csv(
            channels_file,
            sep="\t"
        )


        required = [

            "name",
            "type",
            "soz"

        ]


        for c in required:

            if c not in channels.columns:

                raise RuntimeError(
                    f"channels.tsv missing {c}"
                )


        return channels



    # =====================================================
    # SELECT ONLY SEEG CHANNELS
    # =====================================================

    def select_seeg_channels(
        self,
        raw,
        channel_table
    ):


        seeg_table = channel_table[
            channel_table["type"]
            .astype(str)
            .str.upper()
            ==
            "SEEG"
        ]


        if len(seeg_table) == 0:

            raise RuntimeError(
                "No SEEG channels found"
            )


        seeg_names = (
            seeg_table["name"]
            .tolist()
        )


        existing = [

            ch

            for ch in seeg_names

            if ch in raw.ch_names

        ]


        if len(existing) == 0:

            raise RuntimeError(
                "No SEEG channels match EDF channels"
            )


        return seeg_table, existing



    # =====================================================
    # CHANNEL PADDING
    # =====================================================

    def pad_channels(
        self,
        signal,
        labels
    ):


        C,T = signal.shape


        if C > self.max_channels:

            raise RuntimeError(

                f"Channels {C} exceed "
                f"maximum {self.max_channels}"

            )


        padded_signal = torch.zeros(
            (
                self.max_channels,
                T
            ),
            dtype=torch.float32
        )


        padded_labels = torch.zeros(
            (
                self.max_channels,
            ),
            dtype=torch.float32
        )


        channel_mask = torch.zeros(
            (
                self.max_channels,
            ),
            dtype=torch.bool
        )


        padded_signal[:C] = torch.from_numpy(
            signal
        )


        padded_labels[:C] = torch.from_numpy(
            labels
        )


        channel_mask[:C] = True



        return (

            padded_signal,
            padded_labels,
            channel_mask

        )



    # =====================================================
    # GET ITEM
    # =====================================================

    def __getitem__(
        self,
        idx
    ):


        row = self.df.iloc[idx]


        edf_file = Path(
            row["file"]
        )


        channels_file = Path(
            row["channels_file"]
        )



        # ---------------------------------
        # Read EDF
        # ---------------------------------

        raw = mne.io.read_raw_edf(

            edf_file,

            preload=True,

            verbose=False

        )



        fs = float(
            raw.info["sfreq"]
        )



        # ---------------------------------
        # Channel metadata
        # ---------------------------------

        channel_table = self.load_channel_info(
            channels_file
        )



        seeg_table, selected_channels = (
            self.select_seeg_channels(
                raw,
                channel_table
            )
        )



        # ---------------------------------
        # Extract signal
        # ---------------------------------

        raw.pick_channels(
            selected_channels
        )


        start = int(
            row["start_sample"]
        )


        stop = int(
            row["stop_sample"]
        )


        signal = raw.get_data(

            start=start,

            stop=stop

        )



        # signal:
        # [C,T]



        # ---------------------------------
        # SOZ labels
        # ---------------------------------

        label_table = (

            seeg_table
            .set_index("name")
            .loc[selected_channels]

        )


        soz_labels = (

            label_table["soz"]
            .astype(float)
            .values

        )


        # ---------------------------------
        # Padding
        # ---------------------------------

        signal, soz_labels, channel_mask = (

            self.pad_channels(

                signal,

                soz_labels

            )

        )



        return {


            "signal":
                signal,


            "soz_label":
                soz_labels,


            "channel_mask":
                channel_mask,


            "fs":
                torch.tensor(
                    fs,
                    dtype=torch.float32
                ),


            "patient_id":
                row["patient_id"],

        }