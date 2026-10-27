#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
epiSEEG_mayo_fnusa_pathlogical_pipline.py

Full revised EpiSEEG/BioSignal downstream pipeline for pathological event
classification using Mayo Clinic + FNUSA iEEG datasets on Ubuntu/bash server.

Direct paths used from your server:
    Mayo  : /home/ubuntu/IJAZ/Mayo+FUNSA dataet/Dataset_Mayo
    FNUSA : /home/ubuntu/IJAZ/Mayo+FUNSA dataet/BIDS_FNUSA
    Weight: /home/ubuntu/IJAZ/SEEG datast/pretrainingdataset/runs/scale_aligned_brant_edf_stream_real/final_pretrained.pt

Expected segments.csv columns:
    index, anatomy, category_id, channel, electrode_type, institution,
    patient_id, reviewer_id, segment_id, soz, category_name

Modes:
    check_model  -> load pretrained EpiSEEG weight and test forward/representation
    inspect      -> read segments.csv, count data, plot waveform/PSD/spectrogram
    train        -> train/evaluate downstream classifier with patient-level splits
    all          -> run check_model + inspect + train

Recommended first run:
    python epiSEEG_mayo_fnusa_pathlogical_pipline.py --mode check_model

Then:
    python epiSEEG_mayo_fnusa_pathlogical_pipline.py --mode inspect --task three_class --bandpass

Debug training:
    python epiSEEG_mayo_fnusa_pathlogical_pipline.py --mode train --task three_class \
      --protocol combined_5fold --max-per-class 100 --n-splits 2 --seeds 11 \
      --epochs 2 --batch-size 4 --num-workers 0 --bandpass

Full training:
    python epiSEEG_mayo_fnusa_pathlogical_pipline.py --mode train --task three_class \
      --protocol combined_5fold --n-splits 5 --seeds 11 22 33 44 55 \
      --epochs 30 --patience 7 --batch-size 8 --grad-accum-steps 4 \
      --num-workers 2 --bandpass --balanced-sampler
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import random
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.signal as sig

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import label_binarize

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =============================================================================
# Direct default paths from your Ubuntu server
# =============================================================================
DEFAULT_EPISEEG_CKPT = (
    "/home/ubuntu/IJAZ/SEEG datast/pretrainingdataset/runs/"
    "scale_aligned_brant_edf_stream_real/final_pretrained.pt"
)

DEFAULT_MAYO_DIR = "/home/ubuntu/IJAZ/Mayo+FUNSA dataet/Dataset_Mayo"
DEFAULT_FNUSA_DIR = "/home/ubuntu/IJAZ/Mayo+FUNSA dataet/BIDS_FNUSA"
DEFAULT_OUT_DIR = "/home/ubuntu/SEEg results /pathological_event_mayo_fnusa_episeeg"

# Your real EpiSEEG pretrained model settings
DEFAULT_MODEL_KWARGS = {
    "d_model": 192,
    "temporal_layers": 3,
    "channel_layers": 2,
    "n_heads": 6,
    "dim_feedforward": 768,
    "dropout": 0.1,
    "max_channels": 64,
    "t_align": 8,
    "f_align": 128,
    "f_max": 128.0,
    "conv_channels": 24,
    "stft_chunk_size": 512,
}

CLASS_3 = ["physiological", "pathological", "artifact"]
CLASS_4 = ["physiological", "pathological", "artifact", "powerline"]
CLASS_BIN = ["non_pathological", "pathological"]


# =============================================================================
# Utility functions
# =============================================================================
def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def mkdir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_json(obj: Any, path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def norm_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize segments.csv column names:
    - lower case
    - remove spaces
    - fix common spelling variants
    """
    df = df.copy()
    df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

    rename_map = {
        "instition": "institution",
        "institition": "institution",
        "institution_id": "institution",
        "segemnt_id": "segment_id",
        "segement_id": "segment_id",
        "segment": "segment_id",
        "soz_status": "soz",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    return df


def first_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    for name in names:
        if name in df.columns:
            return name
    return None


# =============================================================================
# Checkpoint/model loading
# =============================================================================
def find_state_dict(ckpt: Any) -> Tuple[Dict[str, torch.Tensor], str]:
    """
    Supports:
      - {'model': state_dict}
      - {'model_state_dict': state_dict}
      - {'state_dict': state_dict}
      - direct top-level state_dict
    """
    if isinstance(ckpt, dict):
        for key in ["model", "model_state_dict", "state_dict", "net", "encoder", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                if any(torch.is_tensor(v) for v in ckpt[key].values()):
                    return ckpt[key], key

        if any(torch.is_tensor(v) for v in ckpt.values()):
            return ckpt, "top_level_state_dict"

    raise RuntimeError("No valid state_dict found in checkpoint.")


def strip_module(sd: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Remove 'module.' prefix if the model was saved from DataParallel/DDP.
    """
    return {
        k.replace("module.", "", 1) if k.startswith("module.") else k: v
        for k, v in sd.items()
    }


def count_params(sd: Dict[str, torch.Tensor]) -> Dict[str, Any]:
    total = 0
    floating = 0
    modules = defaultdict(int)

    for name, value in sd.items():
        if torch.is_tensor(value):
            n = value.numel()
            total += n
            if value.dtype.is_floating_point:
                floating += n
            modules[name.replace("module.", "").split(".")[0]] += n

    return {
        "total_tensors_parameters_buffers": int(total),
        "floating_parameters_buffers": int(floating),
        "approx_fp32_mb": floating * 4 / 1024**2,
        "approx_fp16_mb": floating * 2 / 1024**2,
        "module_counts": dict(sorted(modules.items(), key=lambda x: x[1], reverse=True)),
    }


def build_model(args, device: torch.device) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Build EpiSEEG model from model_scale_brant.ScaleAlignedBrantMAE
    and load final_pretrained.pt.
    """
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd, src = find_state_dict(ckpt)
    sd = strip_module(sd)

    cfg = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    data_cfg = cfg.get("data", {}) if isinstance(cfg, dict) else {}
    train_cfg = cfg.get("train", {}) if isinstance(cfg, dict) else {}

    module = importlib.import_module(args.model_module)
    model_cls = getattr(module, args.model_class)
    signature = inspect.signature(model_cls.__init__)

    # Start with checkpoint config if available; otherwise use real EpiSEEG defaults.
    candidates = {
        "d_model": model_cfg.get("d_model", DEFAULT_MODEL_KWARGS["d_model"]),
        "embed_dim": model_cfg.get("d_model", DEFAULT_MODEL_KWARGS["d_model"]),
        "embedding_dim": model_cfg.get("d_model", DEFAULT_MODEL_KWARGS["d_model"]),
        "hidden_dim": model_cfg.get("d_model", DEFAULT_MODEL_KWARGS["d_model"]),

        "temporal_layers": model_cfg.get("temporal_layers", DEFAULT_MODEL_KWARGS["temporal_layers"]),
        "temporal_encoder_layers": model_cfg.get("temporal_layers", DEFAULT_MODEL_KWARGS["temporal_layers"]),

        "channel_layers": model_cfg.get("channel_layers", DEFAULT_MODEL_KWARGS["channel_layers"]),
        "channel_encoder_layers": model_cfg.get("channel_layers", DEFAULT_MODEL_KWARGS["channel_layers"]),

        "n_heads": model_cfg.get("n_heads", DEFAULT_MODEL_KWARGS["n_heads"]),
        "num_heads": model_cfg.get("n_heads", DEFAULT_MODEL_KWARGS["n_heads"]),

        "dim_feedforward": model_cfg.get("dim_feedforward", DEFAULT_MODEL_KWARGS["dim_feedforward"]),
        "mlp_dim": model_cfg.get("dim_feedforward", DEFAULT_MODEL_KWARGS["dim_feedforward"]),

        "dropout": model_cfg.get("dropout", DEFAULT_MODEL_KWARGS["dropout"]),

        "t_align": model_cfg.get("t_align", DEFAULT_MODEL_KWARGS["t_align"]),
        "f_align": model_cfg.get("f_align", DEFAULT_MODEL_KWARGS["f_align"]),
        "f_max": model_cfg.get("f_max", DEFAULT_MODEL_KWARGS["f_max"]),
        "conv_channels": model_cfg.get("conv_channels", DEFAULT_MODEL_KWARGS["conv_channels"]),
        "stft_chunk_size": model_cfg.get("stft_chunk_size", DEFAULT_MODEL_KWARGS["stft_chunk_size"]),

        "max_channels": data_cfg.get("max_channels", DEFAULT_MODEL_KWARGS["max_channels"]),
        "num_channels": data_cfg.get("max_channels", DEFAULT_MODEL_KWARGS["max_channels"]),

        "context_sec": data_cfg.get("context_sec", 60),
        "patch_sec": data_cfg.get("patch_sec", 1),
        "mask_ratio": train_cfg.get("mask_ratio", 0.4),
    }

    if args.model_kwargs:
        candidates.update(json.loads(args.model_kwargs))

    # Only pass kwargs accepted by your model constructor.
    kwargs = {k: v for k, v in candidates.items() if k in signature.parameters}

    print("\n================ MODEL BUILD ================")
    print("MODEL:", args.model_module + "." + args.model_class)
    print("CONSTRUCTOR:", signature)
    print("ACCEPTED KWARGS:", json.dumps(kwargs, indent=2))

    model = model_cls(**kwargs)
    missing, unexpected = model.load_state_dict(sd, strict=args.strict_load)

    print("CHECKPOINT:", args.checkpoint)
    print("STATE_DICT_SOURCE:", src)
    print("MISSING_KEYS:", len(missing))
    print("UNEXPECTED_KEYS:", len(unexpected))
    if missing:
        print("FIRST_MISSING:", missing[:10])
    if unexpected:
        print("FIRST_UNEXPECTED:", unexpected[:10])

    model.to(device)

    info = {
        "checkpoint": args.checkpoint,
        "state_dict_source": src,
        "accepted_kwargs": kwargs,
        "missing_keys": missing,
        "unexpected_keys": unexpected,
        "param_counts": count_params(sd),
        "config_if_available": cfg,
    }

    return model, info


def call_with_fs(fn, x: torch.Tensor, fs: int) -> Any:

    """

    Robust EpiSEEG forward caller.



    Your model forward requires:

        forward(x, fs, channel_mask)



    For Mayo/FNUSA each sample is single-channel, so channel_mask is valid channel = 1.

    """

    errors = []



    if x.ndim >= 3:

        b = x.shape[0]

        c = x.shape[1]

    else:

        b = x.shape[0]

        c = 1



    masks = []

    for n_ch in [c, 64]:

        masks.append(torch.ones((b, n_ch), dtype=torch.bool, device=x.device))

        masks.append(torch.ones((b, n_ch), dtype=torch.float32, device=x.device))



    for channel_mask in masks:

        call_patterns = [

            lambda: fn(x, fs, channel_mask),

            lambda: fn(x, fs=fs, channel_mask=channel_mask),

            lambda: fn(x, channel_mask=channel_mask, fs=fs),

            lambda: fn(x, sampling_rate=fs, channel_mask=channel_mask),

            lambda: fn(x, sample_rate=fs, channel_mask=channel_mask),

        ]



        for call in call_patterns:

            try:

                return call()

            except Exception as e:

                errors.append(repr(e))



    # Fallbacks without channel mask

    for kwargs in [{"fs": fs}, {"sampling_rate": fs}, {"sample_rate": fs}, {}]:

        try:

            return fn(x, **kwargs)

        except Exception as e:

            errors.append(repr(e))



    try:

        return fn(x, fs)

    except Exception as e:

        errors.append(repr(e))



    raise RuntimeError("All EpiSEEG forward signatures failed:\n" + "\n".join(errors[:20]))

def select_tensor(out: Any) -> torch.Tensor:
    """
    Select feature tensor from model output.
    """
    if torch.is_tensor(out):
        if out.ndim >= 2:
            return out
        raise RuntimeError("Only scalar/1D tensor returned.")

    if isinstance(out, dict):
        preferred = [
            "Z", "z", "features", "feature", "embedding", "embeddings",
            "latent", "repr", "representation", "encoder_output",
            "last_hidden_state",
        ]
        for key in preferred:
            if key in out and torch.is_tensor(out[key]) and out[key].ndim >= 2:
                return out[key]

        for key, value in out.items():
            if torch.is_tensor(value) and value.ndim >= 2:
                return value

        raise RuntimeError("No representation tensor in output dict keys=" + str(list(out.keys())))

    if isinstance(out, (tuple, list)):
        for value in out:
            if torch.is_tensor(value) and value.ndim >= 2:
                return value

    raise RuntimeError("Unsupported model output type: " + str(type(out)))


def pool_representation(z: torch.Tensor) -> torch.Tensor:

    """

    Convert encoder output to one vector per clip: B x D.



    Correct behavior:

    - If output is B x C x N x D, average C and N.

    - If output is B x C x N x T x D, average C, N, and T.

    - Keep only the final feature dimension D.

    """

    if z.ndim == 2:

        return z



    if z.ndim >= 3:

        # Average all dimensions except batch and final feature dimension.

        dims = tuple(range(1, z.ndim - 1))

        return z.mean(dim=dims)



    return z.reshape(z.shape[0], -1)



def extract_rep(encoder: nn.Module, x: torch.Tensor, fs: int) -> torch.Tensor:
    """
    Extract EpiSEEG representation. It first tries feature methods,
    then falls back to model forward().
    """
    errors = []

    for name in [
        "extract_features",
        "forward_features",
        "encode",
        "forward_encoder",
        "get_representation",
        "get_features",
    ]:
        if hasattr(encoder, name):
            try:
                out = call_with_fs(getattr(encoder, name), x, fs)
                return pool_representation(select_tensor(out))
            except Exception as e:
                errors.append(name + ": " + repr(e))

    try:
        out = call_with_fs(encoder, x, fs)
        return pool_representation(select_tensor(out))
    except Exception as e:
        errors.append("forward: " + repr(e))
        raise RuntimeError(
            "Cannot extract EpiSEEG representation. "
            "Add/verify forward_features(), extract_features(), encode(), or forward_encoder().\n"
            + "\n".join(errors)
        )


class EpiSEEGClassifier(nn.Module):
    """
    Pretrained EpiSEEG encoder + small downstream classification head.
    """
    def __init__(self, encoder: nn.Module, fs: int, n_classes: int, dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        self.fs = fs
        self.n_classes = n_classes
        self.dropout = dropout
        self.head = None

    def build_head(self, dim: int, device: torch.device) -> None:
        self.head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(self.dropout),
            nn.Linear(dim, self.n_classes),
        ).to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = extract_rep(self.encoder, x, self.fs)
        if self.head is None:
            raise RuntimeError("Classification head not built.")
        return self.head(r)


def check_model(args) -> None:
    """
    Check final_pretrained.pt and dummy forward pass.
    """
    out = mkdir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    sd, src = find_state_dict(ckpt)
    sd = strip_module(sd)
    counts = count_params(sd)

    print("\n================ EPISEEG CHECKPOINT CHECK ================")
    print("Checkpoint:", args.checkpoint)
    print("State source:", src)
    print("Floating parameters/buffers:", f"{counts['floating_parameters_buffers']:,}")
    print("Approx FP32 MB:", f"{counts['approx_fp32_mb']:.2f}")
    print("Top module counts:")
    for k, v in list(counts["module_counts"].items())[:20]:
        print(f"{k:35s} {v:,}")

    model, info = build_model(args, device)
    model.eval()

    seconds = args.clip_seconds_for_model
    x = torch.randn(1, 1, int(seconds), int(args.target_fs), device=device)

    try:
        with torch.no_grad():
            r = extract_rep(model, x, args.target_fs)

        info["dummy_forward"] = {
            "status": "PASS",
            "input_shape": list(x.shape),
            "representation_shape": list(r.shape),
        }
        print("DUMMY FORWARD PASS: PASS", tuple(x.shape), "->", tuple(r.shape))

    except Exception as e:
        info["dummy_forward"] = {"status": "FAIL", "error": repr(e)}
        print("DUMMY FORWARD PASS: FAIL")
        print(repr(e))

    save_json(info, out / "episeeg_checkpoint_check.json")
    print("Saved:", out / "episeeg_checkpoint_check.json")


# =============================================================================
# Metadata loading
# =============================================================================
def label_norm(x: Any) -> str:

    """

    Convert Mayo/FNUSA category_name into standard labels.



    Plain 'noise' is treated as artifact for the three-class task:

        physiological vs pathological vs artifact



    Explicit power-line labels are treated as powerline.

    """

    s = str(x).strip().lower()

    s = s.replace("-", "_").replace(" ", "_")



    if "phys" in s or "normal" in s or "background" in s:

        return "physiological"



    if "path" in s or "epilept" in s or "spike" in s or "hfo" in s or "seiz" in s:

        return "pathological"



    if "power" in s or "line" in s or "50" in s or "60" in s:

        return "powerline"



    if "noise" in s or "artifact" in s or "artefact" in s or "muscle" in s or "jump" in s or "movement" in s:

        return "artifact"



    return s

def build_index(data_dir: str | Path, out: str | Path, refresh: bool = False) -> Dict[str, str]:
    """
    Index all .mat files under dataset folder.
    """
    data_dir = Path(data_dir)
    out = mkdir(out)
    idx_path = out / (data_dir.name + "_mat_index.json")

    if idx_path.exists() and not refresh:
        return json.loads(idx_path.read_text())

    print("Indexing .mat files:", data_dir)
    idx = {}

    for p in data_dir.rglob("*.mat"):
        idx[p.name] = str(p)
        idx[p.stem] = str(p)
        idx[str(p.relative_to(data_dir))] = str(p)

    idx_path.write_text(json.dumps(idx))
    print("Indexed unique .mat files:", len(set(idx.values())))
    print("Saved index:", idx_path)
    return idx


def resolve_path(row: pd.Series, data_dir: str | Path, idx: Dict[str, str]) -> Optional[str]:
    """
    Resolve MAT file path using segment_id.

    Your segments.csv has segment_id like:
        x000000, x000001, ...

    This function tries:
        x000000
        x000000.mat
        segment_x000000.mat
        x000000_data.mat
    """
    candidates = []

    for col in [
        "segment_id",
        "filename",
        "file_name",
        "file",
        "filepath",
        "file_path",
        "mat_file",
        "path",
        "id",
        "clip",
    ]:
        if col in row.index and pd.notna(row[col]):
            value = str(row[col]).strip()
            if not value:
                continue

            candidates += [
                value,
                value if value.endswith(".mat") else value + ".mat",
                Path(value).name,
                Path(value).stem,
                f"segment_{value}",
                f"segment_{value}.mat",
                f"{value}_data",
                f"{value}_data.mat",
            ]

    for cand in candidates:
        if cand in idx:
            return idx[cand]

        p = Path(cand)
        if p.is_absolute() and p.exists():
            return str(p)

        p2 = Path(data_dir) / cand
        if p2.exists():
            return str(p2)

    return None


def read_dataset(data_dir: str | Path, fallback_institution: str, args, out: Path) -> pd.DataFrame:
    """
    Read Mayo or FNUSA folder.

    Required columns from segments.csv:
        category_name, patient_id, segment_id

    Optional but used for analysis:
        anatomy, channel, electrode_type, institution, reviewer_id, soz
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    segments_files = list(data_dir.rglob("segments.csv"))
    if not segments_files:
        raise FileNotFoundError("segments.csv not found under " + str(data_dir))

    seg_path = segments_files[0]
    print(f"\nReading {fallback_institution} segments.csv:", seg_path)

    df = norm_cols(pd.read_csv(seg_path))
    df["segments_csv"] = str(seg_path)

    cat = first_col(df, ["category_name", "category", "class", "label", "annotation", "event"])
    pat = first_col(df, ["patient_id", "patient", "subject_id", "subject", "participant_id"])
    ch = first_col(df, ["channel", "channel_id", "ch", "ch_name", "name"])
    soz = first_col(df, ["soz", "seizure_onset_zone", "seizure_onset"])
    anat = first_col(df, ["anatomy", "location", "brain_region"])
    elec = first_col(df, ["electrode_type", "electrode", "electrode_name"])
    inst = first_col(df, ["institution", "institute"])
    seg = first_col(df, ["segment_id", "id"])

    if cat is None:
        raise RuntimeError(f"category_name/category column not found. Columns={df.columns.tolist()}")
    if pat is None:
        raise RuntimeError(f"patient_id column not found. Columns={df.columns.tolist()}")
    if seg is None:
        raise RuntimeError(f"segment_id column not found. Columns={df.columns.tolist()}")

    if inst is not None:
        df["institution"] = df[inst].astype(str)
    else:
        df["institution"] = fallback_institution

    # Clean institution labels.
    df["institution"] = df["institution"].astype(str).str.strip()
    df.loc[df["institution"].str.lower().isin(["mayo", "mayoclinic", "mayo_clinic"]), "institution"] = "Mayo"
    df.loc[df["institution"].str.lower().isin(["fnusa", "funsa", "st_annes", "st anne"]), "institution"] = "FNUSA"

    # If CSV institution is missing or weird, use folder fallback.
    bad_inst = df["institution"].str.lower().isin(["", "none", "nan", "null"])
    df.loc[bad_inst, "institution"] = fallback_institution

    df["event_label"] = df[cat].apply(label_norm)
    df["patient_id_norm"] = df["institution"].astype(str) + "_" + df[pat].astype(str)
    df["source_patient_id"] = df[pat].astype(str)
    df["segment_id_norm"] = df[seg].astype(str)

    df["channel_name"] = df[ch].astype(str) if ch else "unknown"
    df["soz_status"] = df[soz].astype(str) if soz else "unknown"
    df["anatomy"] = df[anat].astype(str) if anat else "unknown"
    df["electrode_info"] = df[elec].astype(str) if elec else "unknown"

    if args.task == "three_class":
        keep = CLASS_3
        label_map = {
            "physiological": 0,
            "pathological": 1,
            "artifact": 2,
        }

    elif args.task == "binary":
        keep = ["physiological", "pathological", "artifact"]
        label_map = {
            "physiological": 0,
            "artifact": 0,
            "pathological": 1,
        }

    else:
        keep = CLASS_4
        label_map = {
            "physiological": 0,
            "pathological": 1,
            "artifact": 2,
            "powerline": 3,
        }

    before = len(df)
    df = df[df["event_label"].isin(keep)].copy()

    print(f"{fallback_institution}: kept {len(df)}/{before} rows for task={args.task}")
    print(df["event_label"].value_counts())

    df["label"] = df["event_label"].map(label_map).astype(int)

    idx = build_index(data_dir, out, args.refresh_index)
    df["mat_path"] = df.apply(lambda row: resolve_path(row, data_dir, idx), axis=1)

    missing = int(df["mat_path"].isna().sum())
    if missing:
        print(f"WARNING {fallback_institution}: missing .mat path for {missing} rows; removing.")
        missing_file = Path(out) / f"{fallback_institution}_missing_path_examples.csv"
        df[df["mat_path"].isna()].head(500).to_csv(missing_file, index=False)
        print("Saved missing examples:", missing_file)
        df = df[df["mat_path"].notna()].copy()

    if args.max_per_class > 0:
        df = (
            df.groupby("event_label", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), args.max_per_class), random_state=42))
            .reset_index(drop=True)
        )

    return df.reset_index(drop=True)


def load_all_metadata(args, out: Path) -> pd.DataFrame:
    frames = []

    if args.mayo_dir:
        frames.append(read_dataset(args.mayo_dir, "Mayo", args, out))

    if args.fnusa_dir:
        frames.append(read_dataset(args.fnusa_dir, "FNUSA", args, out))

    if not frames:
        raise ValueError("Provide --mayo-dir and/or --fnusa-dir")

    df = pd.concat(frames, ignore_index=True)
    df.to_csv(Path(out) / "task_metadata.csv", index=False)
    return df


# =============================================================================
# Signal loading/preprocessing
# =============================================================================
def load_mat(path: str | Path) -> np.ndarray:
    """
    Load MATLAB file. Expected variable is 'data'.
    """
    path = Path(path)

    try:
        m = sio.loadmat(path)
        if "data" in m:
            x = m["data"]
        else:
            keys = [k for k in m if not k.startswith("__")]
            if len(keys) != 1:
                raise KeyError(f"data not found in {path}; keys={keys}")
            x = m[keys[0]]

    except NotImplementedError:
        with h5py.File(path, "r") as f:
            if "data" in f:
                x = np.array(f["data"])
            else:
                keys = list(f.keys())
                if len(keys) != 1:
                    raise KeyError(f"data not found in {path}; keys={keys}")
                x = np.array(f[keys[0]])

    return np.asarray(x).squeeze().reshape(-1).astype(np.float32)


def preprocess(x: np.ndarray, args) -> np.ndarray:
    """
    Preprocess one 3-s iEEG clip.

    Default:
        original_fs = 5000
        target_fs   = 1000
        output      = 3 s x 1000 Hz = 3000 samples

    If --clip-seconds-for-model 60 is used, the 3-s clip is repeated to 60 s.
    Use 60 only if your EpiSEEG forward requires 60-s input.
    """
    x = np.asarray(x, dtype=np.float32).reshape(-1)

    if args.original_fs != args.target_fs:
        g = math.gcd(int(args.original_fs), int(args.target_fs))
        x = sig.resample_poly(x, args.target_fs // g, args.original_fs // g).astype(np.float32)

    if args.bandpass:
        nyq = args.target_fs / 2
        hi = min(args.high_freq, nyq - 1)
        if hi > args.low_freq:
            b, a = sig.butter(4, [args.low_freq / nyq, hi / nyq], btype="bandpass")
            x = sig.filtfilt(b, a, x).astype(np.float32)

    if args.notch:
        for f0 in args.notch_freqs:
            if 0 < f0 < args.target_fs / 2:
                b, a = sig.iirnotch(f0 / (args.target_fs / 2), Q=30)
                x = sig.filtfilt(b, a, x).astype(np.float32)

    # Robust normalization.
    med = np.median(x)
    q25, q75 = np.percentile(x, [25, 75])
    scale = q75 - q25
    if scale < 1e-6:
        scale = np.std(x) + 1e-6

    x = np.clip((x - med) / scale, -10, 10).astype(np.float32)

    n = int(args.clip_seconds_for_model * args.target_fs)

    if args.clip_seconds_for_model == 60:
        # Repeat 3-s clip to 60-s only when requested.
        n3 = int(3 * args.target_fs)
        x3 = x[:n3]
        reps = int(np.ceil(n / len(x3)))
        x = np.tile(x3, reps)[:n]
    else:
        if len(x) > n:
            x = x[:n]
        elif len(x) < n:
            x = np.pad(x, (0, n - len(x)))

    return x.astype(np.float32)


# =============================================================================
# Inspection plots
# =============================================================================
def plot_counts(df: pd.DataFrame, out: Path) -> None:
    """
    Save data summary tables and distribution plots.
    """
    pd.DataFrame([{
        "n_clips": len(df),
        "n_patients": df.patient_id_norm.nunique(),
        "n_patient_specific_channels": df[["patient_id_norm", "channel_name"]].drop_duplicates().shape[0],
        "n_institutions": df.institution.nunique(),
    }]).to_csv(out / "overall_summary.csv", index=False)

    df.event_label.value_counts().rename_axis("event_label").reset_index(name="n_clips").to_csv(
        out / "class_counts.csv", index=False
    )
    pd.crosstab(df.institution, df.event_label).to_csv(out / "institution_class_counts.csv")
    pd.crosstab(df.patient_id_norm, df.event_label).to_csv(out / "patient_class_counts.csv")

    df.groupby(["institution", "patient_id_norm"]).agg(
        n_clips=("label", "size"),
        n_classes=("event_label", "nunique"),
        n_channels=("channel_name", "nunique"),
    ).reset_index().to_csv(out / "per_patient_summary.csv", index=False)

    pd.crosstab(df.event_label, df.soz_status).to_csv(out / "label_by_soz.csv")
    pd.crosstab(df.event_label, df.anatomy).to_csv(out / "label_by_anatomy.csv")

    pivot = pd.crosstab(df.event_label, df.institution)
    fig, ax = plt.subplots(figsize=(8, 5), dpi=200)
    pivot.plot(kind="bar", ax=ax)
    ax.set_xlabel("Event class")
    ax.set_ylabel("Number of 3-s clips")
    ax.set_title("Pathological event class distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "class_distribution_by_institution.png", dpi=300)
    fig.savefig(out / "class_distribution_by_institution.pdf")
    plt.close(fig)

    patient = pd.crosstab(df.patient_id_norm, df.event_label)
    patient = patient.loc[patient.sum(axis=1).sort_values().index]
    fig, ax = plt.subplots(figsize=(9, max(5, 0.22 * len(patient))), dpi=200)
    patient.plot(kind="barh", stacked=True, ax=ax)
    ax.set_xlabel("Number of 3-s clips")
    ax.set_ylabel("Patient")
    ax.set_title("Patient-level clip distribution")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "patient_class_distribution.png", dpi=300)
    fig.savefig(out / "patient_class_distribution.pdf")
    plt.close(fig)


def plot_event_examples(df: pd.DataFrame, args, out: Path) -> None:
    """
    For each event class, save:
      - raw waveform
      - preprocessed waveform
      - PSD
      - spectrogram
    """
    ex = mkdir(out / "event_examples")
    rng = np.random.default_rng(args.seeds[0])

    for label in sorted(df.event_label.unique()):
        sub = df[df.event_label == label].reset_index(drop=True)
        if len(sub) == 0:
            continue

        ids = rng.choice(len(sub), size=min(args.samples_per_class, len(sub)), replace=False)

        for j, idx in enumerate(ids, 1):
            raw = load_mat(sub.loc[int(idx), "mat_path"])
            proc = preprocess(raw, args)

            tr = np.arange(len(raw)) / args.original_fs
            tp = np.arange(len(proc)) / args.target_fs

            fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
            ax.plot(tr, raw, lw=0.7)
            ax.set_title(f"{label}: raw 3-s iEEG clip")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Amplitude")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(ex / f"waveform_raw_{label}_{j}.png", dpi=300)
            plt.close(fig)

            fig, ax = plt.subplots(figsize=(9, 4), dpi=200)
            ax.plot(tp, proc, lw=0.8)
            ax.set_title(f"{label}: preprocessed iEEG clip")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Normalized amplitude")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            fig.savefig(ex / f"waveform_preprocessed_{label}_{j}.png", dpi=300)
            plt.close(fig)

            f, pxx = sig.welch(proc, fs=args.target_fs, nperseg=min(len(proc), args.target_fs))
            fig, ax = plt.subplots(figsize=(7, 4.5), dpi=200)
            ax.semilogy(f, pxx + 1e-12, lw=1.2)
            ax.set_xlim(0, min(args.high_freq, args.target_fs / 2))
            ax.set_title(f"{label}: PSD")
            ax.set_xlabel("Frequency (Hz)")
            ax.set_ylabel("PSD")
            fig.tight_layout()
            fig.savefig(ex / f"psd_{label}_{j}.png", dpi=300)
            plt.close(fig)

            nperseg = min(len(proc), int(0.5 * args.target_fs))
            noverlap = min(int(0.4 * args.target_fs), nperseg // 2)
            f, tt, sxx = sig.spectrogram(proc, fs=args.target_fs, nperseg=nperseg, noverlap=noverlap)
            keep = f <= min(args.high_freq, args.target_fs / 2)

            fig, ax = plt.subplots(figsize=(7.4, 4.5), dpi=200)
            im = ax.pcolormesh(tt, f[keep], 10 * np.log10(sxx[keep] + 1e-12), shading="auto")
            ax.set_title(f"{label}: spectrogram")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Frequency (Hz)")
            fig.colorbar(im, ax=ax, label="Power (dB)")
            fig.tight_layout()
            fig.savefig(ex / f"spectrogram_{label}_{j}.png", dpi=300)
            plt.close(fig)

    print("Saved event plots to:", ex)


def inspect_data(args) -> pd.DataFrame:
    """
    Run data inspection and plot examples.
    """
    out = mkdir(args.output_dir)
    df = load_all_metadata(args, out)

    print("\n================ DATA SUMMARY ================")
    print("clips:", len(df))
    print("patients:", df.patient_id_norm.nunique())
    print(df.groupby(["institution", "event_label"]).size())

    plot_counts(df, out)
    plot_event_examples(df, args, out)

    print("Saved analysis outputs to:", out)
    return df


# =============================================================================
# Dataset and dataloader
# =============================================================================
class ClipDataset(Dataset):
    def __init__(self, df: pd.DataFrame, args):
        self.df = df.reset_index(drop=True)
        self.args = args
        self.cache = Path(args.cache_dir) if args.cache_dir else None
        if self.cache:
            mkdir(self.cache)

    def __len__(self) -> int:
        return len(self.df)

    def cache_path(self, row) -> Path:
        safe_patient = str(row.patient_id_norm).replace("/", "_")
        safe_seg = str(row.segment_id_norm).replace("/", "_")
        return self.cache / (
            f"{safe_patient}_{row.event_label}_{safe_seg}_"
            f"fs{self.args.target_fs}_sec{self.args.clip_seconds_for_model}.npy"
        )

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[i]

        if self.cache:
            cp = self.cache_path(row)
            if cp.exists():
                x = np.load(cp)
            else:
                x = preprocess(load_mat(row.mat_path), self.args)
                np.save(cp, x)
        else:
            x = preprocess(load_mat(row.mat_path), self.args)

        n_patches = int(self.args.clip_seconds_for_model)

        samples_per_patch = int(self.args.target_fs)

        needed = n_patches * samples_per_patch



        if len(x) < needed:

            x = np.pad(x, (0, needed - len(x)))

        elif len(x) > needed:

            x = x[:needed]



        # EpiSEEG expects one sample as C x N x L.

        # DataLoader converts it to B x C x N x L.

        # Mayo/FNUSA: C=1, N=3 or 60, L=1000.

        x = x.reshape(1, n_patches, samples_per_patch)



        return {

            "x": torch.tensor(x, dtype=torch.float32),

            "y": torch.tensor(int(row.label), dtype=torch.long),

        }


def make_loader(df: pd.DataFrame, args, train: bool = False) -> DataLoader:
    ds = ClipDataset(df, args)

    if train and args.balanced_sampler:
        y = df.label.values
        counts = np.bincount(y, minlength=args.n_classes)
        w = 1 / np.maximum(counts, 1)
        sample_weights = w[y]
        sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
        return DataLoader(
            ds,
            batch_size=args.batch_size,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=True,
        )

    return DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=train,
        num_workers=args.num_workers,
        pin_memory=True,
    )


# =============================================================================
# Splits and metrics
# =============================================================================
def split_train_val(df: pd.DataFrame, seed: int, val_frac: float = 0.125) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """

    Patient-level train/validation split with class-completeness check.



    It retries different random states until both train and validation contain

    all available classes. This avoids validation folds with missing classes.

    """

    groups = df.patient_id_norm.values

    required = set(df.label.unique().tolist())



    last_train = None

    last_val = None



    for offset in range(100):

        gss = GroupShuffleSplit(n_splits=1, test_size=val_frac, random_state=seed + offset)

        train_idx, val_idx = next(gss.split(df, groups=groups))



        train_df = df.iloc[train_idx].reset_index(drop=True)

        val_df = df.iloc[val_idx].reset_index(drop=True)



        last_train, last_val = train_df, val_df



        train_labels = set(train_df.label.unique().tolist())

        val_labels = set(val_df.label.unique().tolist())



        if required.issubset(train_labels) and required.issubset(val_labels):

            return train_df, val_df



    print("WARNING: Could not create class-complete validation split after 100 retries.")

    print("Required labels:", required)

    print("Train labels:", set(last_train.label.unique().tolist()))

    print("Val labels:", set(last_val.label.unique().tolist()))



    return last_train, last_val



def make_splits(df: pd.DataFrame, protocol: str, n_splits: int, seed: int):
    splits = []

    if protocol in ["combined_5fold", "mayo_5fold", "fnusa_5fold"]:
        d = df.copy()

        if protocol == "mayo_5fold":
            d = d[d.institution == "Mayo"].reset_index(drop=True)

        if protocol == "fnusa_5fold":
            d = d[d.institution == "FNUSA"].reset_index(drop=True)

        k = min(n_splits, d.patient_id_norm.nunique())
        if k < 2:
            raise RuntimeError(f"Not enough patients for {protocol}: {d.patient_id_norm.nunique()}")

        gkf = GroupKFold(n_splits=k)

        for fold, (trainval_idx, test_idx) in enumerate(gkf.split(d, groups=d.patient_id_norm.values), 1):
            trainval = d.iloc[trainval_idx].reset_index(drop=True)
            test = d.iloc[test_idx].reset_index(drop=True)
            train, val = split_train_val(trainval, seed + fold)
            splits.append((train, val, test, f"{protocol}_fold{fold}"))

    elif protocol == "combined_holdout":
        gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
        trainval_idx, test_idx = next(gss.split(df, groups=df.patient_id_norm.values))
        trainval = df.iloc[trainval_idx].reset_index(drop=True)
        test = df.iloc[test_idx].reset_index(drop=True)
        train, val = split_train_val(trainval, seed + 99)
        splits.append((train, val, test, "combined_holdout"))

    elif protocol == "mayo_to_fnusa":
        mayo = df[df.institution == "Mayo"].reset_index(drop=True)
        fnusa = df[df.institution == "FNUSA"].reset_index(drop=True)
        train, val = split_train_val(mayo, seed, 0.1)
        splits.append((train, val, fnusa, "mayo_to_fnusa"))

    elif protocol == "fnusa_to_mayo":
        fnusa = df[df.institution == "FNUSA"].reset_index(drop=True)
        mayo = df[df.institution == "Mayo"].reset_index(drop=True)
        train, val = split_train_val(fnusa, seed, 0.1)
        splits.append((train, val, mayo, "fnusa_to_mayo"))

    else:
        raise ValueError("Unknown protocol: " + protocol)

    return splits


def metrics(y, pred, prob, class_names: List[str]) -> Dict[str, float]:
    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, pred, average="weighted", zero_division=0),
    }

    precision, recall, f1, support = precision_recall_fscore_support(
        y, pred, labels=list(range(len(class_names))), zero_division=0
    )

    for i, class_name in enumerate(class_names):
        out[f"precision_{class_name}"] = precision[i]
        out[f"recall_{class_name}"] = recall[i]
        out[f"f1_{class_name}"] = f1[i]
        out[f"support_{class_name}"] = int(support[i])

    try:
        if len(class_names) == 2:
            out["auroc_macro_ovr"] = roc_auc_score(y, prob[:, 1])
            out["auprc_macro_ovr"] = average_precision_score((y == 1).astype(int), prob[:, 1])
        else:
            y_bin = label_binarize(y, classes=list(range(len(class_names))))
            out["auroc_macro_ovr"] = roc_auc_score(y_bin, prob, average="macro", multi_class="ovr")
            out["auprc_macro_ovr"] = average_precision_score(y_bin, prob, average="macro")
    except Exception:
        out["auroc_macro_ovr"] = np.nan
        out["auprc_macro_ovr"] = np.nan

    return out


def eval_model(model: EpiSEEGClassifier, loader: DataLoader, criterion, device, class_names: List[str]):
    model.eval()
    loss = 0
    ys = []
    preds = []
    probs = []

    with torch.no_grad():
        for batch in loader:
            x = batch["x"].to(device)
            y = batch["y"].to(device)

            logits = model(x)
            loss += criterion(logits, y).item() * len(y)

            p = torch.softmax(logits, dim=1)
            ys += y.cpu().tolist()
            preds += logits.argmax(1).cpu().tolist()
            probs.append(p.cpu().numpy())

    ys = np.array(ys)
    preds = np.array(preds)
    probs = np.concatenate(probs, axis=0)

    m = metrics(ys, preds, probs, class_names)
    m["loss"] = loss / len(loader.dataset)
    return m, ys, preds, probs


def plot_cm(cm: np.ndarray, names: List[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5), dpi=200)
    im = ax.imshow(cm)
    fig.colorbar(im, ax=ax)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion matrix")

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


# =============================================================================
# Training
# =============================================================================
def train_one_split(train, val, test, name: str, seed: int, args, class_names: List[str], device):
    split_dir = mkdir(Path(args.output_dir) / f"seed_{seed}" / name)

    train.to_csv(split_dir / "train_metadata.csv", index=False)
    val.to_csv(split_dir / "val_metadata.csv", index=False)
    test.to_csv(split_dir / "test_metadata.csv", index=False)

    print("\n================ SPLIT ================")
    print("split:", name, "seed:", seed)
    print("train", len(train), "patients", train.patient_id_norm.nunique(), train.event_label.value_counts().to_dict())
    print("val  ", len(val), "patients", val.patient_id_norm.nunique(), val.event_label.value_counts().to_dict())
    print("test ", len(test), "patients", test.patient_id_norm.nunique(), test.event_label.value_counts().to_dict())

    encoder, info = build_model(args, device)
    model = EpiSEEGClassifier(encoder, args.target_fs, len(class_names), args.dropout).to(device)

    if args.freeze_encoder:
        for p in model.encoder.parameters():
            p.requires_grad = False

    train_loader = make_loader(train, args, train=True)
    val_loader = make_loader(val, args, train=False)
    test_loader = make_loader(test, args, train=False)

    batch = next(iter(train_loader))
    x = batch["x"].to(device)

    with torch.no_grad():
        dim = extract_rep(model.encoder, x, args.target_fs).shape[-1]

    model.build_head(int(dim), device)
    print("Representation dim:", int(dim))

    counts = np.bincount(train.label.values, minlength=len(class_names)).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1)
    weights = weights / weights.mean()

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights, dtype=torch.float32, device=device)
    )

    optimizer = torch.optim.AdamW(
        [
            {"params": [p for p in model.encoder.parameters() if p.requires_grad], "lr": args.encoder_lr},
            {"params": model.head.parameters(), "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)
    best = -1
    best_state = None
    bad = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        ytrue = []
        ypred = []
        total = 0
        optimizer.zero_grad(set_to_none=True)
        t0 = time.time()

        for step, batch in enumerate(train_loader, 1):
            xb = batch["x"].to(device)
            yb = batch["y"].to(device)

            with torch.cuda.amp.autocast(enabled=args.amp):
                logits = model(xb)
                loss = criterion(logits, yb) / args.grad_accum_steps

            scaler.scale(loss).backward()

            if step % args.grad_accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            total += loss.item() * args.grad_accum_steps * len(yb)
            ytrue += yb.cpu().tolist()
            ypred += logits.argmax(1).detach().cpu().tolist()

        train_bacc = balanced_accuracy_score(ytrue, ypred)
        val_metrics, _, _, _ = eval_model(model, val_loader, criterion, device, class_names)
        monitor = val_metrics.get(args.monitor, val_metrics["balanced_accuracy"])

        row = {
            "epoch": epoch,
            "train_loss": total / len(train_loader.dataset),
            "train_balanced_accuracy": train_bacc,
            "val_loss": val_metrics["loss"],
            "val_balanced_accuracy": val_metrics["balanced_accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_auroc_macro_ovr": val_metrics.get("auroc_macro_ovr", np.nan),
            "val_auprc_macro_ovr": val_metrics.get("auprc_macro_ovr", np.nan),
            "seconds": time.time() - t0,
        }
        history.append(row)

        print(
            f"epoch={epoch:03d} "
            f"train_loss={row['train_loss']:.4f} "
            f"val_bacc={val_metrics['balanced_accuracy']:.4f} "
            f"val_f1={val_metrics['macro_f1']:.4f} "
            f"monitor={monitor:.4f}"
        )

        if monitor > best:
            best = monitor
            bad = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, split_dir / "best_finetuned.pt")
        else:
            bad += 1
            if args.patience > 0 and bad >= args.patience:
                print("Early stopping")
                break

    pd.DataFrame(history).to_csv(split_dir / "training_history.csv", index=False)

    if best_state:
        model.load_state_dict(best_state)

    test_metrics, y, pred, prob = eval_model(model, test_loader, criterion, device, class_names)
    cm = confusion_matrix(y, pred, labels=list(range(len(class_names))))

    np.save(split_dir / "confusion_matrix.npy", cm)
    plot_cm(cm, class_names, split_dir / "confusion_matrix.png")

    np.savez(
        split_dir / "test_predictions.npz",
        y_true=y,
        y_pred=pred,
        y_prob=prob,
        class_names=np.array(class_names),
    )

    (split_dir / "classification_report.txt").write_text(
        classification_report(y, pred, target_names=class_names, zero_division=0)
    )

    save_json(info, split_dir / "model_checkpoint_info.json")

    result = {
        "seed": seed,
        "split": name,
        "representation_dim": int(dim),
        "train_patients": train.patient_id_norm.nunique(),
        "val_patients": val.patient_id_norm.nunique(),
        "test_patients": test.patient_id_norm.nunique(),
        "train_clips": len(train),
        "val_clips": len(val),
        "test_clips": len(test),
        **{"test_" + k: v for k, v in test_metrics.items()},
    }

    pd.DataFrame([result]).to_csv(split_dir / "test_metrics.csv", index=False)
    return result


def train(args) -> None:
    out = mkdir(args.output_dir)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print("Device:", device)

    df = load_all_metadata(args, out)

    class_names = CLASS_BIN if args.task == "binary" else CLASS_3 if args.task == "three_class" else CLASS_4
    args.n_classes = len(class_names)

    all_results = []

    for seed in args.seeds:
        seed_all(seed)

        for train_df, val_df, test_df, name in make_splits(df, args.protocol, args.n_splits, seed):
            all_results.append(
                train_one_split(train_df, val_df, test_df, name, seed, args, class_names, device)
            )
            pd.DataFrame(all_results).to_csv(out / "fold_metrics.csv", index=False)

    res = pd.DataFrame(all_results)
    cols = [
        c for c in res.columns
        if c.startswith("test_") and pd.api.types.is_numeric_dtype(res[c])
    ]
    summary = res[cols].agg(["mean", "std"]).T.reset_index().rename(columns={"index": "metric"})
    summary.to_csv(out / "summary_mean_std.csv", index=False)

    print("\n================ FINAL SUMMARY ================")
    print(summary)
    print("Saved:", out / "fold_metrics.csv")
    print("Saved:", out / "summary_mean_std.csv")


# =============================================================================
# CLI
# =============================================================================
def parse_args():
    ap = argparse.ArgumentParser()

    ap.add_argument("--mode", choices=["check_model", "inspect", "train", "all"], default="all")

    ap.add_argument("--mayo-dir", default=DEFAULT_MAYO_DIR)
    ap.add_argument("--fnusa-dir", default=DEFAULT_FNUSA_DIR)
    ap.add_argument("--output-dir", default=DEFAULT_OUT_DIR)

    ap.add_argument("--checkpoint", default=DEFAULT_EPISEEG_CKPT)
    ap.add_argument("--model-module", default="model_scale_brant")
    ap.add_argument("--model-class", default="ScaleAlignedBrantMAE")
    ap.add_argument("--model-kwargs", default=json.dumps(DEFAULT_MODEL_KWARGS))
    ap.add_argument("--strict-load", action="store_true")

    ap.add_argument("--original-fs", type=int, default=5000)
    ap.add_argument("--target-fs", type=int, default=1000)
    ap.add_argument("--clip-seconds-for-model", type=int, default=3, choices=[3, 60])
    ap.add_argument("--bandpass", action="store_true")
    ap.add_argument("--low-freq", type=float, default=0.5)
    ap.add_argument("--high-freq", type=float, default=120.0)
    ap.add_argument("--notch", action="store_true")
    ap.add_argument("--notch-freqs", type=float, nargs="+", default=[50.0, 60.0])
    ap.add_argument("--cache-dir", default=None)

    ap.add_argument("--task", choices=["binary", "three_class", "four_class"], default="three_class")
    ap.add_argument(
        "--protocol",
        choices=[
            "combined_holdout",
            "combined_5fold",
            "mayo_5fold",
            "fnusa_5fold",
            "mayo_to_fnusa",
            "fnusa_to_mayo",
        ],
        default="combined_5fold",
    )
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    ap.add_argument("--max-per-class", type=int, default=0)
    ap.add_argument("--refresh-index", action="store_true")

    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum-steps", type=int, default=4)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--encoder-lr", type=float, default=1e-5)
    ap.add_argument("--head-lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--balanced-sampler", action="store_true")
    ap.add_argument("--freeze-encoder", action="store_true")
    ap.add_argument("--monitor", default="auroc_macro_ovr")
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--cpu", action="store_true")

    ap.add_argument("--samples-per-class", type=int, default=3)

    return ap.parse_args()


if __name__ == "__main__":
    args = parse_args()
    mkdir(args.output_dir)
    save_json(vars(args), Path(args.output_dir) / "run_arguments.json")

    if args.mode in ["check_model", "all"]:
        check_model(args)

    if args.mode in ["inspect", "all"]:
        inspect_data(args)

    if args.mode in ["train", "all"]:
        train(args)