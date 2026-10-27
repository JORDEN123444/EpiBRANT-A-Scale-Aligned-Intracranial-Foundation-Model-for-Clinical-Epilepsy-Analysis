
import argparse

import re

from pathlib import Path



import mne

import numpy as np

import pandas as pd

from scipy import signal



mne.set_log_level("WARNING")





INTRACRANIAL_TYPES = {

    "ECOG",

    "SEEG",

    "IEEG",

    "DBS",

}





def canonical_channel_name(name):

    return re.sub(

        r"[^A-Z0-9]",

        "",

        str(name).strip().upper(),

    )





def infer_source(patient_id):

    patient_lower = str(patient_id).lower()



    if "sourcesink" in patient_lower:

        return "SourceSink"



    if "openieeg" in patient_lower:

        return "OpenIEEG"



    if "zurich" in patient_lower:

        return "Zurich"



    return "Other"





def identify_column(dataframe, candidates, required=True):

    lookup = {

        str(column).strip().lower(): column

        for column in dataframe.columns

    }



    for candidate in candidates:

        if candidate.lower() in lookup:

            return lookup[candidate.lower()]



    if required:

        raise RuntimeError(

            "Could not identify any of these columns: {}. "

            "Available columns: {}".format(

                candidates,

                list(dataframe.columns),

            )

        )



    return None





def read_raw_edf(path):

    try:

        return mne.io.read_raw_edf(

            str(path),

            preload=False,

            verbose=False,

        )



    except Exception:

        return mne.io.read_raw_edf(

            str(path),

            preload=False,

            encoding="latin1",

            verbose=False,

        )





def get_matched_intracranial_channels(raw, tsv_path):

    channels = pd.read_csv(

        tsv_path,

        sep="\t",

    )



    name_column = identify_column(

        channels,

        ["name", "channel", "channel_name"],

    )



    type_column = identify_column(

        channels,

        ["type", "channel_type"],

        required=False,

    )



    status_column = identify_column(

        channels,

        ["status"],

        required=False,

    )



    if type_column is None:

        channel_types = pd.Series(

            ["UNKNOWN"] * len(channels),

            index=channels.index,

        )

    else:

        channel_types = (

            channels[type_column]

            .fillna("UNKNOWN")

            .astype(str)

            .str.strip()

            .str.upper()

        )



    if status_column is None:

        statuses = pd.Series(

            ["good"] * len(channels),

            index=channels.index,

        )

    else:

        statuses = (

            channels[status_column]

            .fillna("good")

            .astype(str)

            .str.strip()

            .str.lower()

        )



    good_mask = ~statuses.isin(

        ["bad", "exclude", "excluded"]

    )



    intracranial_mask = channel_types.isin(

        INTRACRANIAL_TYPES

    )



    selected_tsv_names = (

        channels.loc[

            good_mask & intracranial_mask,

            name_column,

        ]

        .astype(str)

        .str.strip()

        .tolist()

    )



    raw_exact_lookup = {

        str(name).strip(): str(name).strip()

        for name in raw.ch_names

    }



    raw_canonical_lookup = {}

    duplicate_canonical_names = set()



    for raw_name in raw.ch_names:

        raw_name = str(raw_name).strip()

        key = canonical_channel_name(raw_name)



        if key in raw_canonical_lookup:

            duplicate_canonical_names.add(key)

        else:

            raw_canonical_lookup[key] = raw_name



    matched_names = []



    for tsv_name in selected_tsv_names:

        tsv_name = str(tsv_name).strip()



        if tsv_name in raw_exact_lookup:

            matched_names.append(

                raw_exact_lookup[tsv_name]

            )

            continue



        key = canonical_channel_name(tsv_name)



        if (

            key in raw_canonical_lookup

            and key not in duplicate_canonical_names

        ):

            matched_names.append(

                raw_canonical_lookup[key]

            )



    # Remove duplicate matches while preserving order.

    matched_names = list(

        dict.fromkeys(matched_names)

    )



    return matched_names





def apply_notch_filter(data, sampling_rate, frequency):

    nyquist = sampling_rate / 2.0



    if frequency >= nyquist:

        return data



    normalized_frequency = frequency / nyquist



    b, a = signal.iirnotch(

        normalized_frequency,

        Q=30.0,

    )



    return signal.filtfilt(

        b,

        a,

        data,

        axis=-1,

    )





def apply_bandpass_filter(

    data,

    sampling_rate,

    low_frequency=0.5,

    high_frequency=100.0,

):

    nyquist = sampling_rate / 2.0



    high_frequency = min(

        high_frequency,

        nyquist * 0.95,

    )



    low_normalized = low_frequency / nyquist

    high_normalized = high_frequency / nyquist



    sos = signal.butter(

        4,

        [low_normalized, high_normalized],

        btype="bandpass",

        output="sos",

    )



    return signal.sosfiltfilt(

        sos,

        data,

        axis=-1,

    )





def resample_exact(

    data,

    original_sampling_rate,

    target_sampling_rate,

):

    original_duration = (

        data.shape[-1]

        / float(original_sampling_rate)

    )



    target_samples = int(

        round(

            original_duration

            * target_sampling_rate

        )

    )



    target_samples = max(

        target_samples,

        1,

    )



    resampled = signal.resample(

        data,

        target_samples,

        axis=-1,

    )



    return np.asarray(

        resampled,

        dtype=np.float32,

    )





def robust_normalize(data):

    median = np.median(

        data,

        axis=1,

        keepdims=True,

    )



    q25 = np.percentile(

        data,

        25,

        axis=1,

        keepdims=True,

    )



    q75 = np.percentile(

        data,

        75,

        axis=1,

        keepdims=True,

    )



    robust_scale = (

        q75 - q25

    ) / 1.349



    standard_deviation = np.std(

        data,

        axis=1,

        keepdims=True,

    )



    robust_scale = np.where(

        robust_scale > 1e-8,

        robust_scale,

        standard_deviation,

    )



    robust_scale = np.where(

        robust_scale > 1e-8,

        robust_scale,

        1.0,

    )



    normalized = (

        data - median

    ) / robust_scale



    normalized = np.nan_to_num(

        normalized,

        nan=0.0,

        posinf=10.0,

        neginf=-10.0,

    )



    normalized = np.clip(

        normalized,

        -10.0,

        10.0,

    )



    return normalized.astype(

        np.float32

    )





def pad_channels(

    data,

    channel_names,

    maximum_channels,

):

    number_of_channels = min(

        data.shape[0],

        maximum_channels,

    )



    data = data[

        :number_of_channels

    ]



    channel_names = channel_names[

        :number_of_channels

    ]



    padded = np.zeros(

        (

            maximum_channels,

            data.shape[1],

        ),

        dtype=np.float32,

    )



    padded[

        :number_of_channels

    ] = data



    channel_mask = np.zeros(

        maximum_channels,

        dtype=np.float32,

    )



    channel_mask[

        :number_of_channels

    ] = 1.0



    padded_channel_names = np.array(

        channel_names

        + ["PAD"] * (

            maximum_channels

            - number_of_channels

        ),

        dtype="U64",

    )



    return (

        padded,

        channel_mask,

        padded_channel_names,

        number_of_channels,

    )





def safe_filename(value):

    return re.sub(

        r"[^A-Za-z0-9_.-]+",

        "_",

        str(value),

    )





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--manifest",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    parser.add_argument(

        "--target_fs",

        type=int,

        default=250,

    )



    parser.add_argument(

        "--window_sec",

        type=int,

        default=60,

    )



    parser.add_argument(

        "--filter_margin_sec",

        type=float,

        default=5.0,

    )



    parser.add_argument(

        "--max_channels",

        type=int,

        default=128,

    )



    args = parser.parse_args()



    manifest_path = Path(args.manifest)

    out_dir = Path(args.out_dir)



    segment_root = (

        out_dir / "preprocessed_segments"

    )



    segment_root.mkdir(

        parents=True,

        exist_ok=True,

    )



    manifest = pd.read_csv(

        manifest_path

    )



    required_columns = [

        "patient_name",

        "label",

        "numeric_label",

        "split",

        "saved_edf_path",

        "saved_channels_tsv",

    ]



    missing_columns = [

        column

        for column in required_columns

        if column not in manifest.columns

    ]



    if missing_columns:

        raise RuntimeError(

            "Missing manifest columns: {}".format(

                missing_columns

            )

        )



    output_rows = []

    failed_rows = []



    for record_index, record in manifest.iterrows():

        patient_id = str(

            record["patient_name"]

        )



        state = str(

            record["label"]

        ).strip().lower()



        numeric_label = int(

            record["numeric_label"]

        )



        original_split = str(

            record["split"]

        )



        pair_id = str(

            record.get("pair_id", "")

        )



        edf_path = Path(

            str(record["saved_edf_path"])

        )



        tsv_path = Path(

            str(record["saved_channels_tsv"])

        )



        print("\n" + "=" * 100)

        print(

            "[{}/{}] Patient={} | state={}"

            .format(

                record_index + 1,

                len(manifest),

                patient_id,

                state,

            )

        )

        print("EDF:", edf_path)

        print("=" * 100)



        try:

            raw = read_raw_edf(

                edf_path

            )



            sampling_rate = float(

                raw.info["sfreq"]

            )



            duration_sec = float(

                raw.n_times / sampling_rate

            )



            matched_channel_names = (

                get_matched_intracranial_channels(

                    raw,

                    tsv_path,

                )

            )



            if not matched_channel_names:

                raise RuntimeError(

                    "No EDFTSV matched "

                    "intracranial channels."

                )



            matched_channel_names = (

                matched_channel_names[

                    :args.max_channels

                ]

            )



            number_of_windows = int(

                np.floor(

                    duration_sec

                    / args.window_sec

                )

            )



            if number_of_windows < 1:

                raise RuntimeError(

                    "Recording is shorter than one "

                    "complete {}-second window."

                    .format(args.window_sec)

                )



            print(

                "Sampling rate:",

                sampling_rate,

            )



            print(

                "Duration:",

                duration_sec,

            )



            print(

                "Matched channels:",

                len(matched_channel_names),

            )



            print(

                "Complete windows:",

                number_of_windows,

            )



            patient_segment_dir = (

                segment_root

                / safe_filename(patient_id)

            )



            patient_segment_dir.mkdir(

                parents=True,

                exist_ok=True,

            )



            for window_index in range(

                number_of_windows

            ):

                window_start_sec = (

                    window_index

                    * args.window_sec

                )



                window_end_sec = (

                    window_start_sec

                    + args.window_sec

                )



                read_start_sec = max(

                    0.0,

                    window_start_sec

                    - args.filter_margin_sec,

                )



                read_end_sec = min(

                    duration_sec,

                    window_end_sec

                    + args.filter_margin_sec,

                )



                read_start_sample = int(

                    round(

                        read_start_sec

                        * sampling_rate

                    )

                )



                read_end_sample = int(

                    round(

                        read_end_sec

                        * sampling_rate

                    )

                )



                read_end_sample = min(

                    read_end_sample,

                    raw.n_times,

                )



                data = raw.get_data(

                    picks=matched_channel_names,

                    start=read_start_sample,

                    stop=read_end_sample,

                )



                data = np.asarray(

                    data,

                    dtype=np.float64,

                )



                data = np.nan_to_num(

                    data,

                    nan=0.0,

                    posinf=0.0,

                    neginf=0.0,

                )



                # Apply both filters to every dataset so that

                # line frequency cannot identify data source.

                data = apply_notch_filter(

                    data,

                    sampling_rate,

                    50.0,

                )



                data = apply_notch_filter(

                    data,

                    sampling_rate,

                    60.0,

                )



                data = apply_bandpass_filter(

                    data,

                    sampling_rate,

                    low_frequency=0.5,

                    high_frequency=100.0,

                )



                data = resample_exact(

                    data,

                    sampling_rate,

                    args.target_fs,

                )



                crop_start = int(

                    round(

                        (

                            window_start_sec

                            - read_start_sec

                        )

                        * args.target_fs

                    )

                )



                expected_window_samples = int(

                    args.window_sec

                    * args.target_fs

                )



                crop_end = (

                    crop_start

                    + expected_window_samples

                )



                window_data = data[

                    :,

                    crop_start:crop_end,

                ]



                if (

                    window_data.shape[1]

                    < expected_window_samples

                ):

                    missing_samples = (

                        expected_window_samples

                        - window_data.shape[1]

                    )



                    window_data = np.pad(

                        window_data,

                        (

                            (0, 0),

                            (0, missing_samples),

                        ),

                        mode="constant",

                    )



                elif (

                    window_data.shape[1]

                    > expected_window_samples

                ):

                    window_data = window_data[

                        :,

                        :expected_window_samples,

                    ]



                window_data = robust_normalize(

                    window_data

                )



                (

                    padded_data,

                    channel_mask,

                    padded_channel_names,

                    valid_channel_count,

                ) = pad_channels(

                    window_data,

                    matched_channel_names,

                    args.max_channels,

                )



                # Foundation model format:

                # [channels, 60 one-second patches, samples per second]

                model_input = padded_data.reshape(

                    args.max_channels,

                    args.window_sec,

                    args.target_fs,

                )



                segment_id = (

                    "{}_window_{:04d}"

                    .format(

                        safe_filename(patient_id),

                        window_index,

                    )

                )



                segment_path = (

                    patient_segment_dir

                    / "{}.npz".format(

                        segment_id

                    )

                )



                np.savez_compressed(

                    segment_path,

                    x=model_input.astype(

                        np.float32

                    ),

                    channel_mask=channel_mask,

                    channel_names=(

                        padded_channel_names

                    ),

                    patient_id=patient_id,

                    state=state,

                    label=numeric_label,

                    original_fs=sampling_rate,

                    target_fs=args.target_fs,

                    start_sec=window_start_sec,

                    end_sec=window_end_sec,

                )



                output_rows.append({

                    "patient_id": patient_id,

                    "state": state,

                    "label": numeric_label,

                    "original_split": (

                        original_split

                    ),

                    "split": original_split,

                    "pair_id": pair_id,

                    "source_dataset": (

                        infer_source(patient_id)

                    ),

                    "edf_path": str(edf_path),

                    "raw_file": str(edf_path),

                    "channels_tsv": str(tsv_path),

                    "segment_id": segment_id,

                    "window_index": int(

                        window_index

                    ),

                    "start_sec": float(

                        window_start_sec

                    ),

                    "end_sec": float(

                        window_end_sec

                    ),

                    "original_fs": float(

                        sampling_rate

                    ),

                    "target_fs": int(

                        args.target_fs

                    ),

                    "sampling_rate": int(

                        args.target_fs

                    ),

                    "context_sec": int(

                        args.window_sec

                    ),

                    "n_valid_channels": int(

                        valid_channel_count

                    ),

                    "segment_path": str(

                        segment_path

                    ),

                    "cache_path": str(

                        segment_path

                    ),

                    # Aliases retained for compatibility.

                    "npz_path": str(

                        segment_path

                    ),

                    "preprocessed_path": str(

                        segment_path

                    ),

                })



            raw.close()



        except Exception as error:

            failed_rows.append({

                "patient_id": patient_id,

                "state": state,

                "edf_path": str(edf_path),

                "error": repr(error),

            })



            print("FAILED:", repr(error))



    segment_manifest = pd.DataFrame(

        output_rows

    )



    segment_manifest_path = (

        out_dir / "segment_manifest.csv"

    )



    segment_manifest.to_csv(

        segment_manifest_path,

        index=False,

    )



    failure_path = (

        out_dir / "preprocessing_failures.csv"

    )



    pd.DataFrame(

        failed_rows

    ).to_csv(

        failure_path,

        index=False,

    )



    print("\n" + "=" * 100)

    print("PREPROCESSING SUMMARY")

    print("=" * 100)



    print(

        "Successfully created segments:",

        len(segment_manifest),

    )



    print(

        "Failed recordings:",

        len(failed_rows),

    )



    if not segment_manifest.empty:

        print("\nSegments by class:")

        print(

            segment_manifest[

                "state"

            ].value_counts()

        )



        print("\nSegments by patient:")

        print(

            segment_manifest.groupby(

                ["patient_id", "state"]

            ).size().to_string()

        )



        example = np.load(

            segment_manifest.iloc[0][

                "segment_path"

            ]

        )



        print(

            "\nExample signal shape:",

            example["x"].shape,

        )



        print(

            "Example channel-mask shape:",

            example["channel_mask"].shape,

        )



    print("\nSaved:")

    print(segment_manifest_path)

    print(failure_path)



    if failed_rows:

        raise RuntimeError(

            "{} recordings failed preprocessing."

            .format(len(failed_rows))

        )





if __name__ == "__main__":

    main()

