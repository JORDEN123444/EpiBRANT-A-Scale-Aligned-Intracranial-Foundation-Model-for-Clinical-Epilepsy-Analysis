import argparse
import inspect
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import torch
from models import build_model
from omegaconf import OmegaConf
from scipy import signal
from scipy.signal import resample_poly
RAW_COLUMNS = [
    "edf_path",
    "raw_edf_path",
    "recording_path",
    "raw_path",
    "edf_file",
    "file_path",
]

ARRAY_COLUMNS = [
    "segment_path",
    "npz_path",
    "signal_path",
    "array_path",
    "cache_path",
]

START_COLUMNS = [
    "start_sec",
    "segment_start_sec",
    "window_start_sec",
    "offset_sec",
    "start_time",
]

FS_COLUMNS = [
    "sampling_rate",
    "target_fs",
    "sfreq",
    "fs",
    "sample_rate",
]

PATIENT_COLUMNS = [
    "patient_id",
    "patient",
    "subject_id",
    "subject",
]

NON_INTRACRANIAL = re.compile(
    r"(ECG|EKG|EMG|EOG|RESP|TRIG|MARK|ANNOT|EVENT|PULSE|PHOTIC|DC\d*)",
    re.IGNORECASE,
)


def normalise_columns(df):
    return df.rename(
        columns={
            column: re.sub(
                r"[^a-z0-9]+",
                "_",
                str(column).strip().lower(),
            ).strip("_")
            for column in df.columns
        }
    )


def first_value(row, candidates, default=None):
    for column in candidates:
        if column in row.index and not pd.isna(row[column]):
            value = row[column]

            if str(value).strip():
                return value

    return default


def existing_path(row, candidates):
    for column in candidates:
        if column not in row.index or pd.isna(row[column]):
            continue

        path = Path(str(row[column]))

        if path.exists():
            return path

    return None


def state_and_label(row):
    if "state" in row.index and not pd.isna(row["state"]):
        state = str(row["state"]).strip().lower()

        state = {
            "wake": "awake",
            "waking": "awake",
            "asleep": "sleep",
            "sleeping": "sleep",
            "0": "awake",
            "0.0": "awake",
            "1": "sleep",
            "1.0": "sleep",
        }.get(state, state)

        if state in {"awake", "sleep"}:
            return state, int(state == "sleep")

    if "label" in row.index:
        label = int(float(row["label"]))
        return ("sleep" if label == 1 else "awake"), label

    raise RuntimeError("No valid sleep/awake label.")


def canonical_channel(name):
    return re.sub(
        r"[^A-Z0-9]+",
        "",
        str(name).upper(),
    )


def find_sidecar(raw_path, suffix):
    candidates = []

    for parent in [raw_path.parent, raw_path.parent.parent]:
        if parent.exists():
            candidates.extend(
                parent.glob(f"*{suffix}")
            )

    return sorted(candidates)[0] if candidates else None


def load_good_channels(raw_path, raw):
    channels_tsv = find_sidecar(
        raw_path,
        "_channels.tsv",
    )

    if channels_tsv is None:
        picks = [
            index
            for index, name in enumerate(raw.ch_names)
            if not NON_INTRACRANIAL.search(name)
        ]

        return picks, None

    channels = pd.read_csv(
        channels_tsv,
        sep="\t",
    )

    channels.columns = [
        str(column).strip().lower()
        for column in channels.columns
    ]

    allowed_types = {
        "seeg",
        "ecog",
        "ieeg",
        "dbs",
    }

    good_names = set()

    for _, row in channels.iterrows():
        channel_type = str(
            row.get("type", "")
        ).strip().lower()

        status = str(
            row.get("status", "good")
        ).strip().lower()

        name = canonical_channel(
            row.get("name", "")
        )

        if (
            channel_type in allowed_types
            and status not in {"bad", "excluded"}
        ):
            good_names.add(name)

    picks = [
        index
        for index, name in enumerate(raw.ch_names)
        if canonical_channel(name) in good_names
    ]

    return picks, channels_tsv


def load_raw_segment(raw_path, start_sec):
    suffix = raw_path.suffix.lower()

    if suffix in {".edf", ".bdf"}:
        raw = mne.io.read_raw_edf(
            str(raw_path),
            preload=False,
            verbose="ERROR",
        )
    elif suffix == ".fif":
        raw = mne.io.read_raw_fif(
            str(raw_path),
            preload=False,
            verbose="ERROR",
        )
    else:
        raise RuntimeError(
            f"Unsupported raw format: {raw_path}"
        )

    fs = float(raw.info["sfreq"])

    picks, channels_tsv = load_good_channels(
        raw_path,
        raw,
    )

    if not picks:
        raise RuntimeError(
            f"No valid intracranial channels in {raw_path}"
        )

    start = max(
        0,
        int(round(start_sec * fs)),
    )

    stop = min(
        raw.n_times,
        start + int(round(60 * fs)),
    )

    data = raw.get_data(
        picks=picks,
        start=start,
        stop=stop,
    ).astype(np.float64)

    names = [
        raw.ch_names[index]
        for index in picks
    ]

    return data, fs, names, channels_tsv


def load_cached_array(path, row):
    if path.suffix.lower() == ".npy":
        data = np.load(
            path,
            allow_pickle=False,
        )

        archive_metadata = {}

    elif path.suffix.lower() == ".npz":
        archive_metadata = {}

        with np.load(
            path,
            allow_pickle=False,
        ) as archive:
            selected_key = None

            for key in [
                "x",
                "signal",
                "data",
                "eeg",
            ]:
                if key in archive.files:
                    selected_key = key
                    break

            if selected_key is None:
                raise RuntimeError(
                    f"No signal array in {path}"
                )

            data = np.asarray(
                archive[selected_key]
            )

            for key in [
                "fs",
                "sfreq",
                "sampling_rate",
                "target_fs",
            ]:
                if key in archive.files:
                    archive_metadata["fs"] = float(
                        np.asarray(
                            archive[key]
                        ).squeeze()
                    )

            for key in [
                "channel_names",
                "ch_names",
            ]:
                if key in archive.files:
                    archive_metadata["channel_names"] = [
                        str(value)
                        for value in np.asarray(
                            archive[key]
                        ).tolist()
                    ]

            if "channel_mask" in archive.files:
                archive_metadata["channel_mask"] = np.asarray(
                    archive["channel_mask"]
                ).astype(bool).squeeze()

    else:
        raise RuntimeError(
            f"Unsupported cached signal: {path}"
        )

    data = np.asarray(
        data,
        dtype=np.float64,
    )

    if data.ndim == 3:
        data = data.reshape(
            data.shape[0],
            -1,
        )

    if data.ndim == 1:
        data = data[None, :]

    if data.ndim != 2:
        raise RuntimeError(
            f"Expected channels � time, found {data.shape}"
        )

    mask = archive_metadata.get(
        "channel_mask"
    )

    if mask is not None and len(mask) == data.shape[0]:
        data = data[mask]

    fs = archive_metadata.get("fs")

    if fs is None:
        fs_value = first_value(
            row,
            FS_COLUMNS,
        )

        if fs_value is not None:
            fs = float(fs_value)
        elif data.shape[-1] % 60 == 0:
            fs = data.shape[-1] / 60
        else:
            raise RuntimeError(
                "Cached sampling rate is unknown."
            )

    names = archive_metadata.get(
        "channel_names"
    )

    if names is None or len(names) != data.shape[0]:
        names = [
            f"CH{index + 1:03d}"
            for index in range(data.shape[0])
        ]

    return data, float(fs), names


def parse_contact(name):
    match = re.match(
        r"^(.*?)(\d+)$",
        canonical_channel(name),
    )

    if not match:
        return None

    return match.group(1), int(match.group(2))


def name_based_laplacian(data, names):
    groups = {}

    for index, name in enumerate(names):
        parsed = parse_contact(name)

        if parsed is None:
            continue

        stem, number = parsed

        groups.setdefault(
            stem,
            [],
        ).append(
            (number, index)
        )

    referenced = []
    referenced_names = []

    for group in groups.values():
        group = sorted(group)

        if len(group) < 2:
            continue

        for position, (_, channel_index) in enumerate(group):
            neighbours = []

            if position > 0:
                neighbours.append(
                    group[position - 1][1]
                )

            if position < len(group) - 1:
                neighbours.append(
                    group[position + 1][1]
                )

            if not neighbours:
                continue

            referenced.append(
                data[channel_index]
                - data[neighbours].mean(axis=0)
            )

            referenced_names.append(
                names[channel_index]
            )

    if not referenced:
        return None, None

    return np.stack(referenced), referenced_names


def coordinate_laplacian(data, names, electrodes_path, k=4):
    if electrodes_path is None or not electrodes_path.exists():
        return None, None

    electrodes = pd.read_csv(
        electrodes_path,
        sep="\t",
    )

    electrodes.columns = [
        str(column).strip().lower()
        for column in electrodes.columns
    ]

    required = {
        "name",
        "x",
        "y",
        "z",
    }

    if not required.issubset(electrodes.columns):
        return None, None

    coordinate_map = {}

    for _, row in electrodes.iterrows():
        try:
            coordinates = np.asarray(
                [
                    float(row["x"]),
                    float(row["y"]),
                    float(row["z"]),
                ],
                dtype=float,
            )
        except Exception:
            continue

        if not np.isfinite(coordinates).all():
            continue

        coordinate_map[
            canonical_channel(
                row["name"]
            )
        ] = coordinates

    available = [
        (
            index,
            coordinate_map[
                canonical_channel(name)
            ],
        )
        for index, name in enumerate(names)
        if canonical_channel(name) in coordinate_map
    ]

    if len(available) < 3:
        return None, None

    channel_indices = np.asarray(
        [
            item[0]
            for item in available
        ],
        dtype=int,
    )

    coordinates = np.stack(
        [
            item[1]
            for item in available
        ]
    )

    referenced = []
    referenced_names = []

    for local_index, channel_index in enumerate(channel_indices):
        distances = np.linalg.norm(
            coordinates
            - coordinates[local_index],
            axis=1,
        )

        order = np.argsort(distances)
        neighbours = order[
            1:
            1 + min(k, len(order) - 1)
        ]

        if len(neighbours) == 0:
            continue

        neighbour_channels = channel_indices[
            neighbours
        ]

        referenced.append(
            data[channel_index]
            - data[neighbour_channels].mean(axis=0)
        )

        referenced_names.append(
            names[channel_index]
        )

    if not referenced:
        return None, None

    return np.stack(referenced), referenced_names


def index_laplacian(data, names):
    if data.shape[0] < 2:
        return None, None

    referenced = []

    for index in range(data.shape[0]):
        neighbours = []

        if index > 0:
            neighbours.append(index - 1)

        if index < data.shape[0] - 1:
            neighbours.append(index + 1)

        referenced.append(
            data[index]
            - data[neighbours].mean(axis=0)
        )

    return np.stack(referenced), list(names)


def apply_reference(
    data,
    names,
    raw_path,
    allow_index_fallback,
):
    electrodes_path = (
        find_sidecar(
            raw_path,
            "_electrodes.tsv",
        )
        if raw_path is not None
        else None
    )

    referenced, output_names = coordinate_laplacian(
        data,
        names,
        electrodes_path,
    )

    if referenced is not None:
        return (
            referenced,
            output_names,
            "coordinate_laplacian",
            electrodes_path,
        )

    referenced, output_names = name_based_laplacian(
        data,
        names,
    )

    if referenced is not None:
        return (
            referenced,
            output_names,
            "adjacent_contact_laplacian",
            electrodes_path,
        )

    if allow_index_fallback:
        referenced, output_names = index_laplacian(
            data,
            names,
        )

        if referenced is not None:
            return (
                referenced,
                output_names,
                "index_neighbour_fallback",
                electrodes_path,
            )

    raise RuntimeError(
        "A valid Laplacian/local-contact reference could not be constructed."
    )


def resample_to_2048(data, fs):
    if abs(fs - 2048.0) > 1e-6:
        ratio = Fraction(
            2048.0 / fs
        ).limit_denominator(10000)

        data = resample_poly(
            data,
            up=ratio.numerator,
            down=ratio.denominator,
            axis=-1,
        )

    target = 60 * 2048

    if data.shape[-1] < target:
        data = np.pad(
            data,
            (
                (0, 0),
                (0, target - data.shape[-1]),
            ),
        )
    else:
        data = data[:, :target]

    return data.astype(np.float32)


def official_stft(waveform):
    _, _, zxx = signal.stft(
        waveform,
        fs=2048,
        nperseg=400,
        noverlap=350,
        return_onesided=True,
    )

    zxx = np.abs(zxx)[:40]

    mean = zxx.mean(
        axis=-1,
        keepdims=True,
    )

    std = zxx.std(
        axis=-1,
        keepdims=True,
    )

    std[std == 0] = 1.0

    zxx = (
        zxx - mean
    ) / std

    if zxx.shape[-1] <= 20:
        raise RuntimeError(
            f"STFT sequence is too short: {zxx.shape}"
        )

    zxx = zxx[:, 10:-10]
    zxx = np.nan_to_num(zxx)

    return zxx.T.astype(np.float32)


def safe_torch_load(path):
    kwargs = {"map_location": "cpu"}

    if "weights_only" in inspect.signature(torch.load).parameters:
        kwargs["weights_only"] = False

    return torch.load(path, **kwargs)


def build_brainbert(root, checkpoint_path, device):
    sys.path.insert(
        0,
        str(root),
    )



    checkpoint = safe_torch_load(
        checkpoint_path
    )

    if (
        isinstance(checkpoint, dict)
        and "model_cfg" in checkpoint
    ):
        model_cfg = checkpoint[
            "model_cfg"
        ]
    else:
        model_cfg = OmegaConf.load(
            root
            / "conf/model/masked_tf_model_large.yaml"
        )

    model = build_model(
        model_cfg
    )

    if (
        isinstance(checkpoint, dict)
        and "model" in checkpoint
    ):
        state_dict = checkpoint["model"]
    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "state_dict"
        ]
    else:
        state_dict = checkpoint

    cleaned = {}

    for key, value in state_dict.items():
        for prefix in [
            "module.",
            "model.",
        ]:
            if key.startswith(prefix):
                key = key[len(prefix):]

        cleaned[key] = value

    model_state = model.state_dict()

    compatible = {
        key: value
        for key, value in cleaned.items()
        if (
            key in model_state
            and torch.is_tensor(value)
            and tuple(value.shape)
            == tuple(model_state[key].shape)
        )
    }

    coverage = (
        len(compatible)
        / max(1, len(model_state))
    )

    print(
        "Checkpoint tensor coverage:",
        round(coverage, 4),
    )

    if coverage < 0.80:
        raise RuntimeError(
            "BrainBERT checkpoint coverage is below 80%."
        )

    model.load_state_dict(
        compatible,
        strict=False,
    )

    model.to(device)
    model.eval()

    return model, int(model_cfg.hidden_dim)


@torch.no_grad()
def encode_segment(
    model,
    data_2048,
    device,
    batch_size,
):
    embedding_sum = None
    embedding_count = 0
    batch_specs = []

    def process_batch(specifications):
        nonlocal embedding_sum
        nonlocal embedding_count

        if not specifications:
            return

        tensor = torch.from_numpy(
            np.stack(specifications)
        ).to(device)

        with torch.cuda.amp.autocast(
            enabled=device.type == "cuda"
        ):
            hidden = model(
                tensor,
                src_key_mask=None,
                intermediate_rep=True,
            )

        pooled = hidden.float().mean(
            dim=1
        ).cpu().numpy()

        current_sum = pooled.sum(
            axis=0
        )

        if embedding_sum is None:
            embedding_sum = current_sum
        else:
            embedding_sum += current_sum

        embedding_count += len(pooled)

    chunk_length = 5 * 2048

    for channel_index in range(
        data_2048.shape[0]
    ):
        for start in range(
            0,
            60 * 2048,
            chunk_length,
        ):
            waveform = data_2048[
                channel_index,
                start:start + chunk_length,
            ]

            if len(waveform) != chunk_length:
                continue

            specification = official_stft(
                waveform
            )

            batch_specs.append(
                specification
            )

            if len(batch_specs) >= batch_size:
                process_batch(
                    batch_specs
                )

                batch_specs = []

    process_batch(
        batch_specs
    )

    if embedding_count == 0:
        raise RuntimeError(
            "No valid channel/chunk embeddings were produced."
        )

    embedding = (
        embedding_sum
        / embedding_count
    )

    norm = np.linalg.norm(
        embedding
    )

    if norm > 0:
        embedding = embedding / norm

    return embedding.astype(np.float32), embedding_count


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--brainbert_root",
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--manifest",
        required=True,
    )

    parser.add_argument(
        "--dataset",
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        required=True,
    )

    parser.add_argument(
        "--device",
        default="cuda",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--allow_cached_fallback",
        action="store_true",
    )

    parser.add_argument(
        "--allow_index_fallback",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(
        args.brainbert_root
    )

    checkpoint_path = Path(
        args.checkpoint
    )

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = torch.device(
        args.device
        if (
            args.device == "cpu"
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    model, hidden_dimension = build_brainbert(
        root,
        checkpoint_path,
        device,
    )

    manifest = normalise_columns(
        pd.read_csv(
            args.manifest
        )
    )

    records = []
    failures = []

    for row_index, row in manifest.iterrows():
        try:
            state, label = state_and_label(
                row
            )

            patient_id = str(
                first_value(
                    row,
                    PATIENT_COLUMNS,
                    default=f"patient_{row_index:04d}",
                )
            )

            segment_id = str(
                first_value(
                    row,
                    [
                        "segment_id",
                        "window_id",
                        "sample_id",
                    ],
                    default=f"{args.dataset}_{row_index:06d}",
                )
            )

            raw_path = existing_path(
                row,
                RAW_COLUMNS,
            )

            source_mode = None
            channels_tsv = None

            if raw_path is not None:
                start_sec = float(
                    first_value(
                        row,
                        START_COLUMNS,
                        default=0.0,
                    )
                )

                data, fs, names, channels_tsv = load_raw_segment(
                    raw_path,
                    start_sec,
                )

                source_mode = "native_raw"

            elif args.allow_cached_fallback:
                array_path = existing_path(
                    row,
                    ARRAY_COLUMNS,
                )

                if array_path is None:
                    raise FileNotFoundError(
                        "No native raw or cached signal path."
                    )

                data, fs, names = load_cached_array(
                    array_path,
                    row,
                )

                raw_path = None
                source_mode = (
                    "cached_non_native_fallback"
                )

            else:
                raise FileNotFoundError(
                    "No original raw recording was found. "
                    "Use --allow_cached_fallback only for a pilot analysis."
                )

            usable = (
                np.isfinite(data).all(axis=1)
                & (np.std(data, axis=1) > 1e-12)
            )

            data = data[usable]
            names = [
                name
                for name, keep in zip(
                    names,
                    usable,
                )
                if keep
            ]

            if data.shape[0] < 2:
                raise RuntimeError(
                    "Fewer than two usable channels."
                )

            (
                referenced,
                referenced_names,
                reference_method,
                electrodes_path,
            ) = apply_reference(
                data,
                names,
                raw_path,
                args.allow_index_fallback,
            )

            data_2048 = resample_to_2048(
                referenced,
                fs,
            )

            embedding, encoded_instances = encode_segment(
                model,
                data_2048,
                device,
                args.batch_size,
            )

            safe_id = re.sub(
                r"[^A-Za-z0-9_.-]+",
                "_",
                segment_id,
            )

            embedding_path = (
                output_dir
                / f"{safe_id}.npy"
            )

            np.save(
                embedding_path,
                embedding,
                allow_pickle=False,
            )

            records.append({
                "dataset": args.dataset,
                "model": "BrainBERT",
                "segment_id": segment_id,
                "patient_id": patient_id,
                "state": state,
                "label": label,
                "embedding_path": str(
                    embedding_path
                ),
                "embedding_dimension": int(
                    hidden_dimension
                ),
                "source_mode": source_mode,
                "raw_path": (
                    str(raw_path)
                    if raw_path is not None
                    else ""
                ),
                "original_sampling_rate": fs,
                "brainbert_sampling_rate": 2048,
                "reference_method": reference_method,
                "channels_tsv": (
                    str(channels_tsv)
                    if channels_tsv is not None
                    else ""
                ),
                "electrodes_tsv": (
                    str(electrodes_path)
                    if electrodes_path is not None
                    else ""
                ),
                "input_channels": int(
                    data.shape[0]
                ),
                "referenced_channels": int(
                    referenced.shape[0]
                ),
                "encoded_channel_chunks": int(
                    encoded_instances
                ),
            })

            print(
                f"[{row_index + 1}/{len(manifest)}] "
                f"{segment_id}: embedding={embedding.shape}, "
                f"channels={referenced.shape[0]}, "
                f"reference={reference_method}, "
                f"source={source_mode}"
            )

        except Exception as error:
            failures.append({
                "dataset": args.dataset,
                "row_number": row_index,
                "segment_id": row.get(
                    "segment_id",
                    "",
                ),
                "error": repr(error),
            })

            print(
                f"FAILED row {row_index}: {repr(error)}"
            )

    manifest_path = (
        output_dir.parent
        / f"{args.dataset}_brainbert_embedding_manifest.csv"
    )

    failure_path = (
        output_dir.parent
        / f"{args.dataset}_brainbert_embedding_failures.csv"
    )

    pd.DataFrame(
        records
    ).to_csv(
        manifest_path,
        index=False,
    )

    pd.DataFrame(
        failures
    ).to_csv(
        failure_path,
        index=False,
    )

    print("\nSuccessful:", len(records))
    print("Failed:", len(failures))
    print("Manifest:", manifest_path)
    print("Failures:", failure_path)

    if not records:
        raise RuntimeError(
            "No BrainBERT embeddings were extracted."
        )


if __name__ == "__main__":
    main()
