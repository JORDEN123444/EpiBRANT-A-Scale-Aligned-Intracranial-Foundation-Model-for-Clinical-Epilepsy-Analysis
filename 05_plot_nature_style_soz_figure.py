#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable


# ============================================================
# Utilities
# ============================================================

def infer_split_col(df):
    for c in ["split_internal", "split"]:
        if c in df.columns:
            return c
    raise ValueError("No split column found in manifest. Expected 'split_internal' or 'split'.")


def natural_key(x):
    import re
    parts = re.split(r"(\d+)", str(x))
    return [int(p) if p.isdigit() else p.lower() for p in parts]


def clean_channel_name_series(df):
    if "clean_channel_name" in df.columns:
        s = df["clean_channel_name"].astype(str).fillna("")
        if (s != "").any():
            return s
    if "channel_name" in df.columns:
        return df["channel_name"].astype(str)
    raise ValueError("Neither clean_channel_name nor channel_name exists.")


def attach_manifest_metadata(pred_df, manifest_df):
    """
    Attach metadata by row order.
    This is needed because train/val_segment_predictions.csv often does not include seizure_id.
    """
    pred_df = pred_df.reset_index(drop=True).copy()
    manifest_df = manifest_df.reset_index(drop=True).copy()

    if len(pred_df) != len(manifest_df):
        raise RuntimeError(
            f"Prediction rows ({len(pred_df)}) != manifest rows ({len(manifest_df)})."
        )

    cols_to_attach = [
        "patient_id",
        "patient_name",
        "seizure_id",
        "edf_path",
        "edf_name",
        "target_type",
        "target_rel_start_sec",
        "target_rel_end_sec",
        "segment_start_sec",
        "segment_end_sec",
        "channel_name",
        "clean_channel_name",
        "label",
        "split",
        "split_internal",
    ]

    for c in cols_to_attach:
        if c in manifest_df.columns and c not in pred_df.columns:
            pred_df[c] = manifest_df[c].values

    # If some columns already exist but are missing/empty, overwrite safely
    if "seizure_id" not in pred_df.columns and "seizure_id" in manifest_df.columns:
        pred_df["seizure_id"] = manifest_df["seizure_id"].values

    return pred_df


def load_predictions_with_manifest(run_dir, manifest_csv):
    run_dir = Path(run_dir)
    manifest = pd.read_csv(manifest_csv)
    split_col = infer_split_col(manifest)

    all_parts = []

    for split_name in ["train", "val"]:
        pred_path = run_dir / f"{split_name}_segment_predictions.csv"
        if not pred_path.exists():
            continue

        pred_df = pd.read_csv(pred_path)
        m = manifest[manifest[split_col].astype(str).str.lower() == split_name].copy()

        print(f"Loading: {pred_path}")
        print(f"Prediction rows: {len(pred_df)}")
        print(f"Manifest rows for split: {len(m)}")

        pred_df = attach_manifest_metadata(pred_df, m)
        pred_df["used_prediction_split"] = split_name
        all_parts.append(pred_df)

    if not all_parts:
        raise FileNotFoundError("No train_segment_predictions.csv or val_segment_predictions.csv found.")

    out = pd.concat(all_parts, axis=0, ignore_index=True)
    return out, manifest


def pick_probability_column(df):
    for c in ["prob_soz", "prob", "score", "prediction"]:
        if c in df.columns:
            return c
    raise ValueError("No probability column found. Expected one of: prob_soz, prob, score, prediction.")


# ============================================================
# Aggregation
# ============================================================

def compute_summary_tables(df, prob_col="prob_soz", threshold=0.015, top_k=5, top_n=49):
    """
    df must already contain only early ictal rows (0 to +30 sec).
    """

    df = df.copy()
    df["channel_disp"] = clean_channel_name_series(df)
    df["seizure_order_name"] = df["seizure_id"].astype(str)

    # Order seizures naturally
    seizure_order = sorted(df["seizure_order_name"].unique(), key=natural_key)

    # Time bins: 0,3,6,...,27
    bin_order = sorted(df["target_rel_start_sec"].unique())

    # --------------------------------------------------------
    # Panel A: probability across seizures and bins
    # --------------------------------------------------------
    prob_bin = (
        df.groupby(["seizure_order_name", "target_rel_start_sec", "channel_disp"], as_index=False)[prob_col]
        .mean()
    )

    # --------------------------------------------------------
    # Channel-level summary statistics
    # --------------------------------------------------------
    # Mean probability per channel across all early ictal bins
    mean_prob = (
        df.groupby("channel_disp", as_index=False)[prob_col]
        .mean()
        .rename(columns={prob_col: "mean_probability"})
    )

    # True discharge occurrence rate:
    # fraction of early-ictal bins above threshold
    tmp_occ = df.copy()
    tmp_occ["pred_positive"] = (tmp_occ[prob_col] >= threshold).astype(int)
    occurrence = (
        tmp_occ.groupby("channel_disp", as_index=False)["pred_positive"]
        .mean()
        .rename(columns={"pred_positive": "discharge_occurrence_rate"})
    )

    # Clinical label count (report-derived positive label)
    clinical = (
        df.groupby("channel_disp", as_index=False)["label"]
        .max()
        .rename(columns={"label": "clinical_positive"})
    )

    # Mean prob per seizure for top-k frequency
    per_seiz_ch = (
        df.groupby(["seizure_order_name", "channel_disp"], as_index=False)[prob_col]
        .mean()
    )

    topk_rows = []
    for sid in seizure_order:
        s = per_seiz_ch[per_seiz_ch["seizure_order_name"] == sid].copy()
        s = s.sort_values(prob_col, ascending=False)
        s = s.head(top_k)
        s["topk_hit"] = 1
        topk_rows.append(s[["seizure_order_name", "channel_disp", "topk_hit"]])

    if len(topk_rows) > 0:
        topk_df = pd.concat(topk_rows, axis=0, ignore_index=True)
        onset_freq = (
            topk_df.groupby("channel_disp", as_index=False)["topk_hit"]
            .sum()
            .rename(columns={"topk_hit": "onset_frequency"})
        )
    else:
        onset_freq = pd.DataFrame(columns=["channel_disp", "onset_frequency"])

    # Merge all stats
    rank_df = mean_prob.merge(occurrence, on="channel_disp", how="outer")
    rank_df = rank_df.merge(onset_freq, on="channel_disp", how="left")
    rank_df = rank_df.merge(clinical, on="channel_disp", how="left")
    rank_df["onset_frequency"] = rank_df["onset_frequency"].fillna(0).astype(int)
    rank_df["clinical_positive"] = rank_df["clinical_positive"].fillna(0).astype(int)
    rank_df = rank_df.fillna(0)

    # Ranking rule
    rank_df = rank_df.sort_values(
        by=["onset_frequency", "discharge_occurrence_rate", "mean_probability", "channel_disp"],
        ascending=[False, False, False, True]
    ).reset_index(drop=True)

    top_channels = rank_df["channel_disp"].head(top_n).tolist()

    # Restrict to top channels
    rank_top = rank_df[rank_df["channel_disp"].isin(top_channels)].copy()
    rank_top = rank_top.set_index("channel_disp").loc[top_channels].reset_index()

    # Build panel A matrix
    col_pairs = []
    for sid in seizure_order:
        for b in bin_order:
            col_pairs.append((sid, b))

    A = np.full((len(top_channels), len(col_pairs)), np.nan, dtype=float)

    ch_to_i = {ch: i for i, ch in enumerate(top_channels)}
    pair_to_j = {p: j for j, p in enumerate(col_pairs)}

    for _, r in prob_bin.iterrows():
        ch = r["channel_disp"]
        sid = r["seizure_order_name"]
        b = r["target_rel_start_sec"]
        if ch in ch_to_i and (sid, b) in pair_to_j:
            A[ch_to_i[ch], pair_to_j[(sid, b)]] = float(r[prob_col])

    # Grid matrices for panels B and C
    Bmat, Bch = reshape_channels_to_grid(rank_top["channel_disp"].tolist(),
                                         rank_top["discharge_occurrence_rate"].tolist())
    Cmat, Cch = reshape_channels_to_grid(rank_top["channel_disp"].tolist(),
                                         rank_top["onset_frequency"].tolist())

    # Clinical positive masks for red boxes
    clinical_map = dict(zip(rank_top["channel_disp"], rank_top["clinical_positive"]))

    return {
        "seizure_order": seizure_order,
        "bin_order": bin_order,
        "top_channels": top_channels,
        "rank_top": rank_top,
        "panelA_matrix": A,
        "panelA_col_pairs": col_pairs,
        "panelB_matrix": Bmat,
        "panelB_channels": Bch,
        "panelC_matrix": Cmat,
        "panelC_channels": Cch,
        "clinical_map": clinical_map,
        "n_seizures": len(seizure_order)
    }


def reshape_channels_to_grid(channels, values):
    n = len(channels)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    mat = np.full((nrows, ncols), np.nan, dtype=float)
    ch_grid = np.full((nrows, ncols), "", dtype=object)

    idx = 0
    for i in range(nrows):
        for j in range(ncols):
            if idx < n:
                mat[i, j] = values[idx]
                ch_grid[i, j] = channels[idx]
                idx += 1
    return mat, ch_grid


# ============================================================
# Brain schematic
# ============================================================

def split_prefix_number(ch):
    import re
    m = re.match(r"([A-Za-z]+)(\d+)?", str(ch))
    if not m:
        return str(ch), 1
    prefix = m.group(1)
    num = m.group(2)
    return prefix.upper(), int(num) if num is not None else 1


def approximate_brain_xy(ch):
    """
    Schematic channel-to-brain placement.
    This is NOT true CT/MRI localization.
    It is only for visual schematic display.
    """
    prefix, num = split_prefix_number(ch)

    base_map = {
        "AH": (-0.25, 0.35),
        "A":  (-0.15, 0.20),
        "AC": (-0.05, 0.10),
        "SG": (-0.20, 0.00),
        "X":  (0.10, 0.28),
        "C":  (0.05, 0.15),
        "OF": (0.15, 0.05),
        "T":  (0.00, -0.05),
        "I":  (0.10, -0.20),
        "H":  (0.25, 0.10),
        "B":  (-0.10, 0.05),
        "D":  (-0.20, -0.15),
        "E":  (-0.05, -0.10),
        "F":  (0.25, -0.05),
        "G":  (0.30, -0.15),
        "M":  (0.35, -0.25),
        "O":  (-0.25, 0.05),
        "P":  (0.00, 0.00),
        "U":  (0.15, 0.10),
        "V":  (0.20, 0.25),
        "Y":  (0.35, 0.25),
        "Z":  (-0.10, 0.30),
    }

    x0, y0 = base_map.get(prefix, (0.0, 0.0))

    # gentle spread by contact number
    spread = (num - 1) * 0.015
    x = x0 + 0.03 * math.sin(spread * 4)
    y = y0 - spread

    return x, y


def draw_brain_outline(ax):
    ax.set_aspect("equal")
    ax.axis("off")

    # shadow
    shadow1 = patches.Ellipse((-0.22, 0), width=0.85, height=1.22,
                              facecolor="#bfbfbf", edgecolor="none", alpha=0.25)
    shadow2 = patches.Ellipse((0.22, 0), width=0.85, height=1.22,
                              facecolor="#bfbfbf", edgecolor="none", alpha=0.25)
    ax.add_patch(shadow1)
    ax.add_patch(shadow2)

    # hemispheres
    left = patches.Ellipse((-0.2, 0), width=0.82, height=1.18,
                           facecolor="#e8e8e8", edgecolor="#9a9a9a", lw=1.2, alpha=0.95)
    right = patches.Ellipse((0.2, 0), width=0.82, height=1.18,
                            facecolor="#e8e8e8", edgecolor="#9a9a9a", lw=1.2, alpha=0.95)
    ax.add_patch(left)
    ax.add_patch(right)

    # midline
    ax.plot([0, 0], [-0.58, 0.58], color="#aaaaaa", lw=1.0)

    # suggest gyri
    gyri_y = np.linspace(-0.45, 0.45, 6)
    for yy in gyri_y:
        ax.add_patch(patches.Arc((-0.2, yy), 0.55, 0.16, angle=0,
                                 theta1=200, theta2=340, lw=0.6, color="#c6c6c6"))
        ax.add_patch(patches.Arc((0.2, yy), 0.55, 0.16, angle=0,
                                 theta1=20, theta2=160, lw=0.6, color="#c6c6c6"))

    ax.set_xlim(-0.85, 0.85)
    ax.set_ylim(-0.70, 0.70)


def draw_brain_panel(ax, channels_df, value_col, title, cmap="viridis", vmax=None, top_annotate=6):
    draw_brain_outline(ax)
    ax.set_title(title, fontsize=11, pad=8)

    if len(channels_df) == 0:
        return

    values = channels_df[value_col].values.astype(float)
    if vmax is None:
        vmax = max(np.nanmax(values), 1.0)

    norm = Normalize(vmin=0, vmax=vmax)
    cm = plt.get_cmap(cmap)

    for _, r in channels_df.iterrows():
        ch = r["channel_disp"]
        val = float(r[value_col])
        x, y = approximate_brain_xy(ch)

        ax.scatter(
            x, y,
            s=240,
            color=cm(norm(val)),
            edgecolor="white",
            linewidth=1.2,
            alpha=0.95,
            zorder=5
        )

    # annotate only top few
    top_anno_df = channels_df.head(top_annotate).copy()
    for _, r in top_anno_df.iterrows():
        ch = r["channel_disp"]
        val = float(r[value_col])
        x, y = approximate_brain_xy(ch)
        ax.text(
            x + 0.03, y + 0.02, ch,
            fontsize=8.5, color="#222222",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75),
            zorder=6
        )

    sm = ScalarMappable(norm=norm, cmap=cm)
    cb = plt.colorbar(sm, ax=ax, fraction=0.05, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.outline.set_linewidth(0.6)


# ============================================================
# Plot panels B and C
# ============================================================

def plot_grid_panel(ax, mat, ch_grid, title, cmap, vmin, vmax, clinical_map, value_fmt, cbar_label):
    im = ax.imshow(mat, cmap=cmap, aspect="equal", vmin=vmin, vmax=vmax)
    ax.set_title(title, fontsize=12, pad=10)
    ax.set_xticks([])
    ax.set_yticks([])

    nrows, ncols = mat.shape
    for i in range(nrows + 1):
        ax.axhline(i - 0.5, color="white", lw=1.0)
    for j in range(ncols + 1):
        ax.axvline(j - 0.5, color="white", lw=1.0)

    for i in range(nrows):
        for j in range(ncols):
            ch = ch_grid[i, j]
            if ch == "":
                continue
            val = mat[i, j]
            txt_color = "white" if (not np.isnan(val) and val < (vmin + vmax) / 2) else "#111111"

            label = f"{ch}\n{value_fmt.format(val)}"
            ax.text(j, i, label, ha="center", va="center", fontsize=7.8,
                    color=txt_color, linespacing=0.9)

            if clinical_map.get(ch, 0) == 1:
                rect = patches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    fill=False, edgecolor="red", linewidth=1.5
                )
                ax.add_patch(rect)

    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.ax.tick_params(labelsize=8)
    cb.set_label(cbar_label, fontsize=9)


# ============================================================
# Main figure
# ============================================================

def make_figure(summary, threshold, top_k, out_png, out_pdf=None):
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.linewidth": 0.8,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })

    rank_top = summary["rank_top"]
    A = summary["panelA_matrix"]
    top_channels = summary["top_channels"]
    seizure_order = summary["seizure_order"]
    bin_order = summary["bin_order"]
    Bmat = summary["panelB_matrix"]
    Bch = summary["panelB_channels"]
    Cmat = summary["panelC_matrix"]
    Cch = summary["panelC_channels"]
    clinical_map = summary["clinical_map"]
    n_seizures = summary["n_seizures"]

    fig = plt.figure(figsize=(18, 11), dpi=300)
    gs = fig.add_gridspec(
        nrows=3, ncols=3,
        width_ratios=[2.2, 1.1, 1.1],
        height_ratios=[1.15, 1.0, 1.0],
        wspace=0.35, hspace=0.42
    )

    # --------------------------------------------------------
    # Panel a
    # --------------------------------------------------------
    axA = fig.add_subplot(gs[:, 0])
    imA = axA.imshow(A, cmap="viridis", aspect="auto", vmin=0, vmax=1)

    # channel labels
    axA.set_yticks(np.arange(len(top_channels)))
    axA.set_yticklabels(top_channels)

    # seizure block labels at centers
    block_len = len(bin_order)
    centers = []
    labels = []
    for i, sid in enumerate(seizure_order):
        start = i * block_len
        end = start + block_len - 1
        centers.append((start + end) / 2.0)
        labels.append(f"Seizure {i + 1}")

        if i > 0:
            axA.axvline(start - 0.5, color="white", lw=2.2)

    axA.set_xticks(centers)
    axA.set_xticklabels(labels)
    axA.set_xlabel("Multiple epileptic seizures")
    axA.set_ylabel("Channels")
    axA.set_title("Channel-level predicted probability", pad=10)

    # top axis with 0..30 s bins repeated
    sec_labels = [f"{int(x)}" for x in bin_order]
    sec_ticks = np.arange(A.shape[1])
    axA_top = axA.secondary_xaxis("top")
    axA_top.set_xticks(sec_ticks)
    axA_top.set_xticklabels(sec_labels * n_seizures, rotation=90, fontsize=6)
    axA_top.set_xlabel("Early ictal bins (sec from seizure onset)", fontsize=9)

    cbarA = fig.colorbar(imA, ax=axA, fraction=0.025, pad=0.01)
    cbarA.set_label("Probability", fontsize=9)
    cbarA.ax.tick_params(labelsize=8)

    axA.text(-0.08, 1.02, "a", transform=axA.transAxes, fontsize=16, fontweight="bold")

    # --------------------------------------------------------
    # Panel b
    # --------------------------------------------------------
    axB = fig.add_subplot(gs[0, 1])
    plot_grid_panel(
        axB, Bmat, Bch,
        title="Epileptic discharge occurrence",
        cmap="viridis",
        vmin=0, vmax=1,
        clinical_map=clinical_map,
        value_fmt="{:.2f}",
        cbar_label="Occurrence rate"
    )
    axB.text(-0.12, 1.05, "b", transform=axB.transAxes, fontsize=16, fontweight="bold")

    # --------------------------------------------------------
    # Panel c
    # --------------------------------------------------------
    axC = fig.add_subplot(gs[1, 1])
    plot_grid_panel(
        axC, Cmat, Cch,
        title=f"Onset channel frequency (top-{top_k} per seizure)",
        cmap="viridis",
        vmin=0, vmax=max(1, n_seizures),
        clinical_map=clinical_map,
        value_fmt="{:.0f}",
        cbar_label="Times selected"
    )
    axC.text(-0.12, 1.05, "c", transform=axC.transAxes, fontsize=16, fontweight="bold")

    # --------------------------------------------------------
    # Panel d: two schematic brain panels
    # --------------------------------------------------------
    brain_gs = gs[0:2, 2].subgridspec(2, 1, hspace=0.35)

    axD1 = fig.add_subplot(brain_gs[0, 0])
    discharge_df = rank_top.sort_values(
        ["discharge_occurrence_rate", "mean_probability"], ascending=[False, False]
    ).head(10)
    draw_brain_panel(
        axD1, discharge_df,
        value_col="discharge_occurrence_rate",
        title="Schematic pinpointing of epileptic discharge region",
        cmap="viridis", vmax=1.0, top_annotate=6
    )
    axD1.text(-0.10, 1.06, "d", transform=axD1.transAxes, fontsize=16, fontweight="bold")

    axD2 = fig.add_subplot(brain_gs[1, 0])
    onset_df = rank_top.sort_values(
        ["onset_frequency", "mean_probability"], ascending=[False, False]
    ).head(10)
    draw_brain_panel(
        axD2, onset_df,
        value_col="onset_frequency",
        title="Schematic pinpointing of candidate SOZ contacts",
        cmap="viridis", vmax=max(1, n_seizures), top_annotate=6
    )

    # --------------------------------------------------------
    # Right-lower text box
    # --------------------------------------------------------
    axTxt = fig.add_subplot(gs[2, 1:])
    axTxt.axis("off")

    top10 = rank_top.head(10).copy()

    lines = []
    lines.append("Candidate SOZ / onset contacts")
    lines.append("")
    lines.append(f"Threshold = {threshold:.3f}")
    lines.append("Window = 0 to +30 sec")
    lines.append(f"Top-k onset frequency = {top_k}")
    lines.append(f"Seizures included = {n_seizures}")
    lines.append("")
    lines.append("Top ranked channels:")

    for i, (_, r) in enumerate(top10.iterrows(), start=1):
        lines.append(
            f"{i:>2}. {r['channel_disp']:<6} "
            f"freq={int(r['onset_frequency'])}, "
            f"prob={r['mean_probability']:.3f}, "
            f"occ={r['discharge_occurrence_rate']:.2f}"
        )

    lines.append("")
    lines.append("Red boxes = report-derived earliest ictal-change contacts.")
    lines.append("Brain panel is schematic.")
    lines.append("For true 3D anatomical pinpointing, use electrode CT/MRI coordinates.")

    text = "\n".join(lines)

    axTxt.text(
        0.02, 0.95, text,
        va="top", ha="left", fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.6", fc="#fafafa", ec="#888888", lw=1.0)
    )

    fig.suptitle(
        "Nature-style summary of channel-level SOZ localization",
        fontsize=15, y=0.995
    )

    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    if out_pdf is not None:
        plt.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Plot Nature-style BrainWave-like SOZ summary figure.")
    parser.add_argument("--run_dir", required=True, help="Directory containing train_segment_predictions.csv / val_segment_predictions.csv")
    parser.add_argument("--manifest_csv", required=True, help="Manifest CSV used for training/evaluation")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.015, help="Probability threshold for discharge occurrence")
    parser.add_argument("--top_k", type=int, default=5, help="Top-k channels per seizure for onset frequency")
    parser.add_argument("--top_n", type=int, default=49, help="Top N ranked channels to display")
    parser.add_argument("--prob_col", default=None, help="Probability column, e.g. prob_soz")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred_all, manifest = load_predictions_with_manifest(args.run_dir, args.manifest_csv)

    prob_col = args.prob_col if args.prob_col is not None else pick_probability_column(pred_all)

    # keep only early ictal bins
    if "target_type" not in pred_all.columns:
        raise RuntimeError("target_type column is missing after manifest attachment.")

    early = pred_all[pred_all["target_type"].astype(str) == "early_ictal_onset"].copy()

    if len(early) == 0:
        raise RuntimeError("No early_ictal_onset rows found.")

    print(f"Rows used for figure: {len(early)}")
    print(f"Seizures included: {sorted(early['seizure_id'].unique(), key=natural_key)}")
    print(f"Probability column: {prob_col}")

    summary = compute_summary_tables(
        early,
        prob_col=prob_col,
        threshold=args.threshold,
        top_k=args.top_k,
        top_n=args.top_n
    )

    # save ranking table
    ranking_csv = out_dir / "nature_style_channel_ranking.csv"
    summary["rank_top"].to_csv(ranking_csv, index=False)

    out_png = out_dir / "nature_style_soz_summary.png"
    out_pdf = out_dir / "nature_style_soz_summary.pdf"

    make_figure(
        summary=summary,
        threshold=args.threshold,
        top_k=args.top_k,
        out_png=out_png,
        out_pdf=out_pdf
    )

    print("Saved:")
    print(out_png)
    print(out_pdf)
    print(ranking_csv)


if __name__ == "__main__":
    main()