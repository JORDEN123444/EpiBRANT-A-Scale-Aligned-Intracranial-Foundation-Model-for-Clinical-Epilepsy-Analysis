import ast
import random

import mne
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset


class EpiBRANTPretrainDataset(Dataset):
    """
    Dataset for EpiBRANT-500M self-supervised pretraining.

    Design
    ------
    1. Uses the global pretraining manifest.
    2. Preserves each recording's native sampling frequency.
    3. Selects one random 60-second window during training.
    4. Reads ONLY that 60-second window from EDF.
    5. Does NOT globally resample the signal.
    6. Does NOT perform whole-window normalization.
       Patch normalization is performed later by scale_alignment.py.
    7. Pads the channel dimension to max_channels=288.
    8. Returns a valid-channel mask so padded channels are ignored.
    9. Never silently truncates real SEEG/iEEG contacts.
    10. Returns dataset / subject / session / recording metadata.

    Returned sample
    ---------------
    signal:
        FloatTensor [288, T]

    fs:
        float
        Native EDF sampling rate.

    channel_mask:
        BoolTensor [288]
        True  = valid intracranial channel
        False = padding

    num_valid_channels:
        int

    dataset:
        str

    subject:
        str

    session:
        str

    recording:
        str

    file:
        str

    start_sample:
        int

    start_seconds:
        float
    """

    def __init__(
        self,
        manifest_path,
        window_seconds=60.0,
        patch_seconds=1.0,
        max_channels=288,
        random_window=True,
    ):
        super().__init__()

        self.manifest_path = manifest_path

        self.window_seconds = float(
            window_seconds
        )

        self.patch_seconds = float(
            patch_seconds
        )

        self.max_channels = int(
            max_channels
        )

        self.random_window = bool(
            random_window
        )

        # ----------------------------------------------------
        # Read global manifest
        # ----------------------------------------------------

        self.manifest = pd.read_csv(
            manifest_path
        )

        # Alias for compatibility with older scripts.
        self.df = self.manifest

        # ----------------------------------------------------
        # Required manifest columns
        # ----------------------------------------------------

        required_columns = [
            "dataset",
            "subject_id",
            "file",
            "sampling_rate",
        ]

        for column in required_columns:

            if column not in self.manifest.columns:

                raise ValueError(
                    f"Manifest is missing required column: "
                    f"{column}"
                )

        if len(self.manifest) == 0:

            raise RuntimeError(
                "Global pretraining manifest is empty."
            )

        # ----------------------------------------------------
        # 60 x 1-second patches
        # ----------------------------------------------------

        ratio = (
            self.window_seconds
            / self.patch_seconds
        )

        self.num_patches = int(
            round(ratio)
        )

        if not np.isclose(
            ratio,
            self.num_patches,
        ):
            raise ValueError(
                "window_seconds must contain an integer "
                "number of patch_seconds."
            )

    # ========================================================
    # LENGTH
    # ========================================================

    def __len__(self):

        return len(
            self.manifest
        )

    # ========================================================
    # CHANNEL-NAME PARSER
    # ========================================================

    @staticmethod
    def _parse_channel_names(
        value,
    ):
        """
        Parse the seeg_channel_names field from the manifest.

        Supports:
        - Python list strings
        - semicolon separated strings
        - comma separated strings
        """

        if value is None:
            return []

        if isinstance(
            value,
            (
                list,
                tuple,
            ),
        ):
            return [
                str(x)
                for x in value
            ]

        try:
            if pd.isna(value):
                return []
        except Exception:
            pass

        value = str(
            value
        ).strip()

        if value == "":
            return []

        # ----------------------------------------------------
        # Example:
        # ['A1', 'A2', 'A3']
        # ----------------------------------------------------

        try:

            parsed = ast.literal_eval(
                value
            )

            if isinstance(
                parsed,
                (
                    list,
                    tuple,
                ),
            ):
                return [
                    str(x).strip()
                    for x in parsed
                    if str(x).strip()
                ]

        except Exception:
            pass

        # ----------------------------------------------------
        # Semicolon-separated representation
        # ----------------------------------------------------

        if ";" in value:

            return [
                x.strip()
                for x in value.split(";")
                if x.strip()
            ]

        # ----------------------------------------------------
        # Comma-separated representation
        # ----------------------------------------------------

        if "," in value:

            return [
                x.strip()
                for x in value.split(",")
                if x.strip()
            ]

        return [
            value
        ]

    # ========================================================
    # AUXILIARY CHANNEL FILTER
    # ========================================================

    @staticmethod
    def _is_auxiliary_channel(
        channel_name,
    ):
        """
        Conservative fallback filtering.

        The manifest-provided seeg_channel_names list is preferred.
        This function is only a fallback when that information
        cannot be used.
        """

        name = str(
            channel_name
        ).upper()

        auxiliary_keywords = [

            "ECG",
            "EKG",

            "EMG",
            "EOG",

            "RESP",
            "RESPIR",

            "SPO2",
            "SP02",

            "PLETH",

            "PULSE",

            "STATUS",

            "TRIG",
            "TRIGGER",

            "EVENT",

            "ANNOT",

            "MARK",
        ]

        return any(
            keyword in name
            for keyword in auxiliary_keywords
        )

    # ========================================================
    # SELECT INTRACRANIAL CHANNELS
    # ========================================================

    def _get_seeg_channels(
        self,
        raw,
        row,
    ):
        """
        Select intracranial channels.

        Priority:
            1. seeg_channel_names from global manifest.
            2. Conservative auxiliary-channel exclusion.
        """

        manifest_channels = []

        if (
            "seeg_channel_names"
            in row.index
        ):

            manifest_channels = (
                self._parse_channel_names(
                    row[
                        "seeg_channel_names"
                    ]
                )
            )

        # ----------------------------------------------------
        # Prefer manifest channel selection
        # ----------------------------------------------------

        if len(
            manifest_channels
        ) > 0:

            available_channels = set(
                raw.ch_names
            )

            picks = [
                channel
                for channel
                in manifest_channels
                if channel
                in available_channels
            ]

            if len(picks) > 0:

                return picks

        # ----------------------------------------------------
        # Fallback filtering
        # ----------------------------------------------------

        picks = [

            channel

            for channel
            in raw.ch_names

            if not self._is_auxiliary_channel(
                channel
            )
        ]

        if len(picks) == 0:

            raise RuntimeError(
                "No valid intracranial channels "
                "could be selected."
            )

        return picks

    # ========================================================
    # DETERMINE EXACT 60-SECOND WINDOW
    # ========================================================

    def _choose_window_bounds(
        self,
        total_samples,
        fs,
    ):
        """
        The calculation here intentionally matches
        ScaleAlignmentEmbedding.

        For each sample:

            patch_samples =
                round(fs * patch_seconds)

            required_samples =
                num_patches * patch_samples

        For:
            window_seconds = 60
            patch_seconds  = 1

        num_patches = 60
        """

        patch_samples = int(
            round(
                float(fs)
                * self.patch_seconds
            )
        )

        if patch_samples <= 0:

            raise RuntimeError(
                f"Invalid patch size for fs={fs}"
            )

        required_samples = (
            self.num_patches
            * patch_samples
        )

        total_samples = int(
            total_samples
        )

        if total_samples < required_samples:

            raise RuntimeError(
                "\nRecording is shorter than the required "
                "pretraining window.\n"
                f"Available samples : {total_samples}\n"
                f"Required samples  : {required_samples}\n"
                f"Sampling rate     : {fs}\n"
                f"Patch samples     : {patch_samples}\n"
            )

        max_start = (
            total_samples
            - required_samples
        )

        if (
            self.random_window
            and max_start > 0
        ):

            start = random.randint(
                0,
                max_start
            )

        else:

            # Deterministic validation:
            # always use beginning of recording.
            start = 0

        stop = (
            start
            + required_samples
        )

        return (
            start,
            stop,
            required_samples,
        )

    # ========================================================
    # PAD VARIABLE CHANNEL COUNT
    # ========================================================

    def _pad_channels_with_mask(
        self,
        signal,
    ):
        """
        signal:
            Tensor [C,T]

        Returns
        -------
        padded_signal:
            [max_channels,T]

        channel_mask:
            BoolTensor [max_channels]

        num_valid_channels:
            int

        Important
        ---------
        We NEVER silently truncate real channels.
        """

        if signal.ndim != 2:

            raise ValueError(
                f"Expected [C,T], "
                f"got {tuple(signal.shape)}"
            )

        num_channels = int(
            signal.shape[0]
        )

        num_samples = int(
            signal.shape[1]
        )

        if num_channels <= 0:

            raise RuntimeError(
                "Recording contains zero valid channels."
            )

        if num_channels > self.max_channels:

            raise RuntimeError(
                "\nThe recording contains more valid "
                "intracranial channels than max_channels.\n"
                f"Valid channels : {num_channels}\n"
                f"max_channels   : {self.max_channels}\n\n"
                "Channels will NOT be silently truncated. "
                "Increase max_channels if this occurs."
            )

        padded_signal = torch.zeros(
            (
                self.max_channels,
                num_samples,
            ),
            dtype=torch.float32,
        )

        channel_mask = torch.zeros(
            self.max_channels,
            dtype=torch.bool,
        )

        # ----------------------------------------------------
        # Copy genuine channels
        # ----------------------------------------------------

        padded_signal[
            :num_channels
        ] = signal

        channel_mask[
            :num_channels
        ] = True

        return (
            padded_signal,
            channel_mask,
            num_channels,
        )

    # ========================================================
    # GET SAMPLE
    # ========================================================

    def __getitem__(
        self,
        index,
    ):

        index = int(
            index
        )

        row = self.manifest.iloc[
            index
        ]

        file_path = str(
            row[
                "file"
            ]
        )

        # ====================================================
        # OPEN EDF HEADER ONLY
        #
        # preload=False is essential for actual pretraining.
        # ====================================================

        raw = mne.io.read_raw_edf(
            file_path,
            preload=False,
            verbose="ERROR",
            encoding="latin1",
        )

        try:

            # ================================================
            # NATIVE SAMPLING FREQUENCY
            # ================================================

            fs = float(
                raw.info[
                    "sfreq"
                ]
            )

            if (
                not np.isfinite(fs)
                or fs <= 0
            ):

                raise RuntimeError(
                    f"Invalid EDF sampling rate: {fs}"
                )

            # ================================================
            # OPTIONAL MANIFEST-vs-EDF FS CHECK
            # ================================================

            manifest_fs = float(
                row[
                    "sampling_rate"
                ]
            )

            if np.isfinite(
                manifest_fs
            ):

                if not np.isclose(
                    fs,
                    manifest_fs,
                    rtol=1e-4,
                    atol=1e-3,
                ):

                    raise RuntimeError(
                        "\nSampling-rate mismatch.\n"
                        f"Manifest fs : {manifest_fs}\n"
                        f"EDF fs      : {fs}\n"
                        f"File        : {file_path}"
                    )

            # ================================================
            # SELECT VALID INTRACRANIAL CHANNELS
            # ================================================

            selected_channels = (
                self._get_seeg_channels(
                    raw,
                    row,
                )
            )

            # ================================================
            # SELECT RANDOM 60-SECOND WINDOW
            #
            # IMPORTANT:
            # We choose the bounds BEFORE calling get_data().
            # ================================================

            (
                start_sample,
                stop_sample,
                required_samples,

            ) = self._choose_window_bounds(
                total_samples=raw.n_times,
                fs=fs,
            )

            # ================================================
            # READ ONLY THE REQUIRED 60 SECONDS
            # ================================================

            data = raw.get_data(
                picks=selected_channels,
                start=start_sample,
                stop=stop_sample,
            )

        finally:

            raw.close()

        # ====================================================
        # NUMPY FORMAT
        # ====================================================

        data = np.asarray(
            data,
            dtype=np.float32,
        )

        if data.ndim != 2:

            raise RuntimeError(
                f"EDF data should have shape [C,T], "
                f"got {data.shape}"
            )

        # ====================================================
        # VERIFY EXACT NUMBER OF SAMPLES
        # ====================================================

        if (
            data.shape[1]
            != required_samples
        ):

            raise RuntimeError(
                "\n60-second extraction failed.\n"
                f"Expected samples : {required_samples}\n"
                f"Returned samples : {data.shape[1]}\n"
                f"File             : {file_path}"
            )

        # ====================================================
        # HANDLE NON-FINITE VALUES
        #
        # Very occasional NaN/Inf values should not destroy
        # an entire multi-day pretraining run.
        # ====================================================

        finite_mask = np.isfinite(
            data
        )

        if not finite_mask.all():

            bad_count = int(
                (~finite_mask).sum()
            )

            bad_fraction = (
                bad_count
                / data.size
            )

            # Large corruption should not silently pass.
            if bad_fraction > 0.01:

                raise RuntimeError(
                    "\nToo many non-finite signal values.\n"
                    f"Bad fraction : {bad_fraction:.6f}\n"
                    f"File         : {file_path}"
                )

            # Very sparse corruption:
            # replace individual invalid values with zero.
            data = np.nan_to_num(
                data,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

        # ====================================================
        # IMPORTANT PRETRAINING RULES
        #
        # NO global resampling here.
        # NO full-window z-score normalization here.
        #
        # ScaleAlignmentEmbedding performs normalization
        # independently inside each 1-second patch.
        # ====================================================

        signal = torch.from_numpy(
            np.ascontiguousarray(
                data
            )
        ).float()

        # ====================================================
        # CHANNEL PADDING + VALID-CHANNEL MASK
        # ====================================================

        (
            signal,
            channel_mask,
            num_valid_channels,

        ) = self._pad_channels_with_mask(
            signal
        )

        # ====================================================
        # METADATA
        # ====================================================

        dataset_source = str(
            row[
                "dataset"
            ]
        )

        subject = str(
            row[
                "subject_id"
            ]
        )

        session = ""

        if "session_id" in row.index:

            value = row[
                "session_id"
            ]

            if not pd.isna(
                value
            ):

                session = str(
                    value
                )

        recording = ""

        if "recording_id" in row.index:

            value = row[
                "recording_id"
            ]

            if not pd.isna(
                value
            ):

                recording = str(
                    value
                )

        # ====================================================
        # RETURN
        # ====================================================

        return {

            "signal": signal,

            "fs": fs,

            "channel_mask": channel_mask,

            "num_valid_channels": num_valid_channels,

            "dataset": dataset_source,

            "subject": subject,

            "session": session,

            "recording": recording,

            "file": file_path,

            # window reproducibility metadata

            "start_sample": int(start_sample),

            "start_seconds": float(start_sample / fs),
        }