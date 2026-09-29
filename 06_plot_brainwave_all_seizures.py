#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BrainWave-style SOZ figure plotting
- Uses ALL seizures found in prediction files
- Replaces "prediction across all the seizures" with "Multiple epileptic seizures"
- Produces:
    Panel A: Channel-level predicted SOZ probability
    Panel B: Epileptic discharge occurrence
    Panel C: Onset channel frequency
    Right panel: summary / candidate SOZ box

Expected inputs:
    --run_dir      directory containing train/val/test prediction csv files
    --manifest_csv original or split manifest
    --out_dir      output directory
"""

import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec


# ============================================================
# Helpers
# ============================================================

def pick_split_column(df):
    if "split_internal" in df.columns:
        return "split_internal"
    if "split" in df.columns:
        return "split"
    return None


def load_manifest(run_dir, manifest_csv):
    run_dir = Path(run_dir)
    split_manifest = run_dir / "manifest_with_internal_split.csv"
    if split_manifest.exists():
        print(f"Using split manifest from run_dir: {split_manifest}")
        return pd.read_csv(split_manifest)
    print(f"Using provided manifest: {manifest_csv}")
    return pd.read_csv(manifest_csv)


def attach_manifest_by_row_order(pred_df, manifest_df, split_name):
    """
    Attach manifest metadata by row order for a given split.
    This is useful because train/val prediction files often do not keep seizure_id.
    """
    split_col = pick_split_column(manifest_df)
    if split_col is None:
        raise RuntimeError("No split column found in manifest.")

    mf = manifest_df[manifest_df[split_col].astype(str).str.lower() == split_name.lower()].reset_index(drop=True)
    pred_df = pred_df.reset_index(drop=True)

    if len(pred_df) != len(mf):
        raise RuntimeError(
            f"Prediction rows ({len(pred_df)}) do not match manifest rows ({len(mf)}) for split={split_name}"
        )

    # add columns if missing
    cols_to_add = [
        "patient_id",
        "patient_name",
        "seizure_id",
        "edf_path",
        "edf_name",
        "channel_name",
        "clean_channel_name",
        "segment_start_sec",
        "segment_end_sec",
        "target_rel_start_sec",
        "target_rel_end_sec",
        "target_type",
        "label",
        "positive_onset_label",
    ]

    for c in cols_to_add:
        if c in mf.columns and c not in pred_df.columns:
            pred_df[c] = mf[c].values

    # overwrite helpful fields if absent or empty
    if "clean_channel_name" not in pred_df.columns:
        if "channel_name" in pred_df.columns:
            pred_df["clean_channel_name"] = pred_df["channel_name"]
    if "edf_name" not in pred_df.columns and "edf_path" in pred_df.columns:
        pred_df["edf_name"] = pred_df["edf_path"].apply(lambda x: Path(str(x)).name)

    return pred_df


def load_all_predictions(run_dir, manifest_df):
    run_dir = Path(run_dir)
    frames = []

    for split_name in ["train", "val", "test"]:
        pred_path = run_dir / f"{split_name}_segment_predictions.csv"
        if pred_path.exists():
            print(f"Loading: {pred_path}")
            pred_df = pd.read_csv(pred_path)
            pred_df = attach_manifest_by_row_order(pred_df, manifest_df, split_name)
            pred_df["prediction_split_file"] = split_name
            print(f"  rows: {len(pred_df)}")
            frames.append(pred_df)

    if not frames:
        raise RuntimeError("No train/val/test prediction csv files found in run_dir.")

    out = pd.concat(frames, axis=0, ignore_index=True)
    return out


def reshape_to_grid(values, labels, ncols=7):
    """
    Turn 1D values into a rectangular grid for Panels B/C.
    """
    n = len(values)
    nrows = math.ceil(n / ncols)

    value_grid = np.full((nrows, ncols), np.nan, dtype=float)
    label_grid = np.full((nrows, ncols), "", dtype=object)

    for i, (v, lab) in enumerate(zip(values, labels)):
        r = i // ncols
        c = i % ncols
        value_grid[r, c] = v
        label_grid[r, c] = lab

    return value_grid, label_grid


def plot_labeled_grid(ax, value_grid, label_grid, title, cmap="viridis",
                      vmin=None, vmax=None, value_fmt=None, cbar_label=None,
                      fontsize_label=8, fontsize_value=7):
    im = ax.imshow(value_grid, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_title(title, fontsize=13, fontweight="bold", pad=8)
    ax.set_xticks([])
    ax.set_yticks([])

    nrows, ncols = value_grid.shape

    # cell borders
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color="white", lw=0.6)
    for j in range(ncols + 1):
        ax.axvline(j - 0.5, color="white", lw=0.6)

    for r in range(nrows):
        for c in range(ncols):
            if label_grid[r, c] == "":
                continue
            v = value_grid[r, c]
            ax.text(c, r - 0.10, str(label_grid[r, c]),
                    ha="center", va="center", fontsize=fontsize_label,
                    color="white", fontweight="bold")
            if not np.isnan(v):
                txt = f"{v:.2f}" if value_fmt is None else value_fmt.format(v)
                ax.text(c, r + 0.20, txt,
                        ha="center", va="center", fontsize=fontsize_value,
                        color="white")

    return im


# ============================================================
# Core figure building
# ============================================================

def make_brainwave_style_figure(
    df,
    out_path,
    threshold=0.015,
    top_n=None,
    prob_col="prob_soz",
    panel_c_mode="auto",
    top_k_pred_freq=5,
):
    """
    Build BrainWave-style figure for one patient using ALL seizures.
    """

    # Keep only early ictal windows for actual SOZ analysis
    if "target_type" in df.columns:
        ictal = df[df["target_type"] == "early_ictal_onset"].copy()
    else:
        # fallback
        ictal = df[(df["target_rel_start_sec"] >= 0) & (df["target_rel_start_sec"] < 30)].copy()

    if len(ictal) == 0:
        raise RuntimeError("No early_ictal_onset rows found.")

    if "clean_channel_name" not in ictal.columns:
        ictal["clean_channel_name"] = ictal["channel_name"]

    # sort seizures by their first time if possible
    seizure_order_df = (
        ictal.groupby("seizure_id", as_index=False)["segment_start_sec"]
        .min()
        .sort_values("segment_start_sec")
    )
    seizure_order = seizure_order_df["seizure_id"].tolist()

    # 10 early ictal bins normally: 0,3,...,27
    bin_order = sorted(ictal["target_rel_start_sec"].dropna().unique().tolist())

    # ---------------------------------------------
    # Panel A matrix: channels x (all seizure bins)
    # ---------------------------------------------
    panelA = (
        ictal.groupby(["clean_channel_name", "seizure_id", "target_rel_start_sec"], as_index=False)[prob_col]
        .mean()
    )

    # candidate stats per channel
    mean_prob_per_channel = panelA.groupby("clean_channel_name")[prob_col].mean()

    # occurrence:
    # a channel "occurs" in a seizure if its max early-ictal probability >= threshold
    occurrence = (
        panelA.groupby(["clean_channel_name", "seizure_id"])[prob_col]
        .max()
        .reset_index()
    )
    occurrence["occur"] = (occurrence[prob_col] >= threshold).astype(int)
    occurrence_rate = occurrence.groupby("clean_channel_name")["occur"].mean()

    # onset frequency:
    # if clinical labels exist, use them
    onset_freq = None
    if panel_c_mode in ["auto", "clinical"]:
        if "positive_onset_label" in ictal.columns:
            tmp = (
                ictal.groupby(["clean_channel_name", "seizure_id"])["positive_onset_label"]
                .max()
                .reset_index()
            )
            onset_freq = tmp.groupby("clean_channel_name")["positive_onset_label"].sum()
            if onset_freq.sum() == 0:
                onset_freq = None

    # fallback: predicted top-k frequency
    if onset_freq is None:
        panel_c_mode = "predicted_topk"
        seizure_means = (
            panelA.groupby(["seizure_id", "clean_channel_name"], as_index=False)[prob_col]
            .mean()
        )

        hit_list = []
        for sid in seizure_order:
            sub = seizure_means[seizure_means["seizure_id"] == sid].copy()
            sub = sub.sort_values(prob_col, ascending=False).head(top_k_pred_freq)
            sub["hit"] = 1
            hit_list.append(sub[["clean_channel_name", "hit"]])

        hit_df = pd.concat(hit_list, axis=0, ignore_index=True)
        onset_freq = hit_df.groupby("clean_channel_name")["hit"].sum()

    # build ranking table
    rank_df = pd.DataFrame({
        "clean_channel_name": sorted(panelA["clean_channel_name"].unique())
    })
    rank_df["mean_prob"] = rank_df["clean_channel_name"].map(mean_prob_per_channel).fillna(0.0)
    rank_df["discharge_occurrence"] = rank_df["clean_channel_name"].map(occurrence_rate).fillna(0.0)
    rank_df["onset_frequency"] = rank_df["clean_channel_name"].map(onset_freq).fillna(0.0)

    rank_df = rank_df.sort_values(
        ["onset_frequency", "discharge_occurrence", "mean_prob", "clean_channel_name"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    if top_n is not None and top_n > 0:
        rank_df = rank_df.head(top_n).copy()

    channel_order = rank_df["clean_channel_name"].tolist()

    # pivot Panel A
    panelA = panelA[panelA["clean_channel_name"].isin(channel_order)].copy()
    panelA["clean_channel_name"] = pd.Categorical(panelA["clean_channel_name"], categories=channel_order, ordered=True)
    panelA["seizure_id"] = pd.Categorical(panelA["seizure_id"], categories=seizure_order, ordered=True)
    panelA["target_rel_start_sec"] = pd.Categorical(panelA["target_rel_start_sec"], categories=bin_order, ordered=True)

    panelA_pivot = panelA.pivot_table(
        index="clean_channel_name",
        columns=["seizure_id", "target_rel_start_sec"],
        values=prob_col,
        aggfunc="mean"
    )

    # ensure full ordered columns
    full_cols = [(sid, b) for sid in seizure_order for b in bin_order]
    panelA_pivot = panelA_pivot.reindex(index=channel_order, columns=pd.MultiIndex.from_tuples(full_cols))
    panelA_matrix = panelA_pivot.values

    # Panel B values
    B_vals = rank_df["discharge_occurrence"].values
    B_labels = rank_df["clean_channel_name"].values
    B_grid, B_lab = reshape_to_grid(B_vals, B_labels, ncols=7)

    # Panel C values
    C_vals = rank_df["onset_frequency"].values
    C_labels = rank_df["clean_channel_name"].values
    C_grid, C_lab = reshape_to_grid(C_vals, C_labels, ncols=7)

    # explanation text
    top_show = min(10, len(rank_df))
    lines = []
    lines.append("Candidate SOZ / onset contacts")
    lines.append("")
    lines.append(f"Threshold = {threshold:.3f}")
    lines.append("Window = 0 to +30 sec")
    lines.append("")
    lines.append("Top ranked channels:")
    for i in range(top_show):
        row = rank_df.iloc[i]
        lines.append(
            f"{i+1}. {row['clean_channel_name']}   "
            f"freq={int(row['onset_frequency'])}, "
            f"prob={row['mean_prob']:.3f}"
        )
    lines.append("")
    if panel_c_mode == "clinical":
        lines.append("Onset frequency = report-derived")
        lines.append("earliest ictal-change contacts.")
    else:
        lines.append("Onset frequency = predicted")
        lines.append(f"top-{top_k_pred_freq} channel count.")
    lines.append("")
    lines.append("3D brain pinpointing requires")
    lines.append("electrode coordinates from CT/MRI.")
    explanation = "\n".join(lines)

    # ============================================================
    # Plot
    # ============================================================
    fig = plt.figure(figsize=(18, 10), dpi=300)
    gs = gridspec.GridSpec(
        2, 3,
        width_ratios=[2.4, 1.6, 0.9],
        height_ratios=[1.0, 1.0],
        wspace=0.45,
        hspace=0.50
    )

    # ------------------------
    # Panel A
    # ------------------------
    axA = fig.add_subplot(gs[:, 0])
    imA = axA.imshow(panelA_matrix, aspect="auto", cmap="viridis", vmin=0, vmax=1)

    axA.set_title("Channel-level predicted probability", fontsize=14, fontweight="bold")
    axA.set_xlabel("Multiple epileptic seizures", fontsize=12)   # <-- replaced text here
    axA.set_ylabel("Channels", fontsize=12)

    axA.set_yticks(np.arange(len(channel_order)))
    axA.set_yticklabels(channel_order, fontsize=8)

    # no S1/S2/S3 labels
    axA.set_xticks([])

    # seizure separators
    n_bins = len(bin_order)
    for k in range(1, len(seizure_order)):
        axA.axvline(k * n_bins - 0.5, color="white", lw=2)

    cbarA = fig.colorbar(imA, ax=axA, fraction=0.025, pad=0.012)
    cbarA.set_label("Probability", fontsize=11)

    axA.text(-0.10, 1.02, "a", transform=axA.transAxes, fontsize=16, fontweight="bold")

    # ------------------------
    # Panel B
    # ------------------------
    axB = fig.add_subplot(gs[0, 1])
    imB = plot_labeled_grid(
        axB,
        B_grid,
        B_lab,
        title="Epileptic discharge occurrence",
        cmap="viridis",
        vmin=0,
        vmax=1,
        value_fmt="{:.2f}",
        fontsize_label=8,
        fontsize_value=7,
    )
    cbarB = fig.colorbar(imB, ax=axB, fraction=0.040, pad=0.02)
    cbarB.set_label("Occurrence rate", fontsize=10)

    axB.text(-0.14, 1.04, "b", transform=axB.transAxes, fontsize=16, fontweight="bold")

    # ------------------------
    # Panel C
    # ------------------------
    axC = fig.add_subplot(gs[1, 1])
    imC = plot_labeled_grid(
        axC,
        C_grid,
        C_lab,
        title="Onset channel frequency",
        cmap="viridis",
        vmin=0,
        vmax=max(1, np.nanmax(C_grid)),
        value_fmt="{:.0f}",
        fontsize_label=8,
        fontsize_value=7,
    )
    cbarC = fig.colorbar(imC, ax=axC, fraction=0.040, pad=0.02)
    cbarC.set_label("Times as onset site", fontsize=10)

    axC.text(-0.14, 1.04, "c", transform=axC.transAxes, fontsize=16, fontweight="bold")

    # ------------------------
    # Right explanation panel
    # ------------------------
    axText = fig.add_subplot(gs[:, 2])
    axText.axis("off")
    axText.text(
        0.02, 0.98, explanation,
        ha="left", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.6", fc="whitesmoke", ec="gray")
    )

    plt.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure: {out_path}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True, type=str)
    parser.add_argument("--manifest_csv", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--threshold", default=0.015, type=float)
    parser.add_argument("--top_n", default=49, type=int)
    parser.add_argument("--prob_col", default="prob_soz", type=str)
    parser.add_argument("--patient_id", default=None, type=str,
                        help="If given, plot only this patient. Otherwise, plot one figure per patient.")
    parser.add_argument("--panel_c_mode", default="auto", choices=["auto", "clinical", "predicted_topk"])
    parser.add_argument("--top_k_pred_freq", default=5, type=int)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = load_manifest(args.run_dir, args.manifest_csv)
    pred_df = load_all_predictions(args.run_dir, manifest_df)

    if "patient_id" not in pred_df.columns:
        raise RuntimeError("patient_id column not found after attaching manifest.")

    patient_ids = sorted(pred_df["patient_id"].dropna().unique().tolist())
    if args.patient_id is not None:
        patient_ids = [args.patient_id]

    print("Patients to plot:", patient_ids)

    for pid in patient_ids:
        sub = pred_df[pred_df["patient_id"] == pid].copy()
        if len(sub) == 0:
            print(f"Skipping {pid}: no rows.")
            continue

        patient_name = str(sub["patient_name"].iloc[0]) if "patient_name" in sub.columns else pid
        safe_name = patient_name.replace("/", "_").replace(" ", "_")
        out_path = out_dir / f"brainwave_style_all_seizures_{pid}_{safe_name}.png"

        print(f"\nPlotting patient: {pid} / {patient_name}")
        print(f"Rows: {len(sub)}")
        print(f"Seizures found: {sorted(sub['seizure_id'].dropna().unique().tolist())}")

        make_brainwave_style_figure(
            df=sub,
            out_path=out_path,
            threshold=args.threshold,
            top_n=args.top_n,
            prob_col=args.prob_col,
            panel_c_mode=args.panel_c_mode,
            top_k_pred_freq=args.top_k_pred_freq,
        )


if __name__ == "__main__":
    main()