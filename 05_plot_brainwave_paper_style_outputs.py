import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle


def clean_channel_name(ch):
    s = str(ch).strip().upper()
    s = s.replace("", "-")
    s = re.sub(r"^(EEG|POL)\s+", "", s)
    s = s.replace(" ", "")
    s = re.sub(r"-?REF$", "", s)
    return s


def natural_channel_key(ch):
    ch = str(ch)
    m = re.match(r"^([A-Z]+)(\d+)$", ch)
    if m:
        return (m.group(1), int(m.group(2)))
    return (ch, 9999)


def find_prob_col(df):
    for c in ["prob_soz", "prob", "probability", "y_prob", "score"]:
        if c in df.columns:
            return c
    raise RuntimeError("No probability column found. Available columns: " + str(df.columns.tolist()))


def load_manifest(run_dir, manifest_csv):
    run_manifest = Path(run_dir) / "manifest_with_internal_split.csv"
    if run_manifest.exists():
        print("Using run manifest:", run_manifest)
        return pd.read_csv(run_manifest)
    print("Using provided manifest:", manifest_csv)
    man = pd.read_csv(manifest_csv)
    if "split_internal" not in man.columns:
        man["split_internal"] = man["split"]
    return man


def load_predictions_with_metadata(run_dir, manifest_csv):
    run_dir = Path(run_dir)
    manifest = load_manifest(run_dir, manifest_csv).copy()
    manifest = manifest.loc[:, ~manifest.columns.duplicated()].copy()

    if "clean_channel_name" not in manifest.columns:
        manifest["clean_channel_name"] = manifest["channel_name"].map(clean_channel_name)

    parts = []

    for split_name in ["train", "val", "test"]:
        p = run_dir / f"{split_name}_segment_predictions.csv"
        if not p.exists():
            continue

        pred = pd.read_csv(p)
        pred = pred.loc[:, ~pred.columns.duplicated()].copy()
        pred["prediction_split_file"] = split_name

        man_split = manifest[manifest["split_internal"].astype(str) == split_name].copy().reset_index(drop=True)
        pred = pred.reset_index(drop=True)

        print("\nLoading:", p)
        print("Prediction rows:", len(pred))
        print("Manifest rows for split:", len(man_split))

        if "seizure_id" not in pred.columns:
            if len(pred) != len(man_split):
                raise RuntimeError(
                    f"Cannot attach metadata for {split_name}: "
                    f"prediction rows={len(pred)} but manifest rows={len(man_split)}"
                )

            meta_cols = [
                "seizure_id",
                "raw_file",
                "channel_name",
                "clean_channel_name",
                "segment_start_sec",
                "segment_end_sec",
                "target_rel_start_sec",
                "target_rel_end_sec",
                "target_type",
                "label",
                "split_internal",
            ]

            pred = pd.concat([man_split[meta_cols], pred], axis=1)
            pred = pred.loc[:, ~pred.columns.duplicated()].copy()
            print("Attached metadata by row order.")

        if "clean_channel_name" not in pred.columns:
            pred["clean_channel_name"] = pred["channel_name"].map(clean_channel_name)

        parts.append(pred)

    if not parts:
        raise RuntimeError("No prediction files found. Run evaluate_soz.py first.")

    pred = pd.concat(parts, ignore_index=True)
    pred = pred.loc[:, ~pred.columns.duplicated()].copy()
    return pred, manifest


def make_channel_grid(channels, values, title, value_label, highlight_channels=None, cmap="viridis", vmin=0, vmax=1):
    highlight_channels = set(highlight_channels or [])

    # Arrange channels into a compact grid.
    n = len(channels)
    n_cols = 7
    n_rows = int(np.ceil(n / n_cols))

    mat = np.full((n_rows, n_cols), np.nan)
    labels = [["" for _ in range(n_cols)] for __ in range(n_rows)]

    val_map = dict(zip(channels, values))

    for idx, ch in enumerate(channels):
        r = idx // n_cols
        c = idx % n_cols
        mat[r, c] = val_map.get(ch, np.nan)
        labels[r][c] = ch

    return mat, labels, n_rows, n_cols


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--manifest_csv", required=True)
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--threshold", type=float, default=0.015)
    parser.add_argument("--top_n", type=int, default=49)
    parser.add_argument("--analysis_start", type=float, default=0.0)
    parser.add_argument("--analysis_end", type=float, default=30.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pred, manifest = load_predictions_with_metadata(run_dir, args.manifest_csv)
    prob_col = find_prob_col(pred)

    pred = pred[pred["seizure_id"].isin(["S1", "S2", "S3"])].copy()

    early = pred[
        (pred["target_rel_start_sec"] >= args.analysis_start) &
        (pred["target_rel_end_sec"] <= args.analysis_end)
    ].copy()

    if early.empty:
        raise RuntimeError("No early ictal rows found.")

    print("\nRows used:", len(early))
    print("Seizures:", sorted(early["seizure_id"].unique()))
    print("Probability column:", prob_col)

    # Main channel summary tables.
    A_tbl = (
        early.groupby(["seizure_id", "clean_channel_name"], as_index=False)[prob_col]
        .mean()
        .rename(columns={prob_col: "mean_predicted_probability"})
    )

    B_tbl = (
        early.assign(is_discharge=(early[prob_col] >= args.threshold).astype(int))
        .groupby("clean_channel_name", as_index=False)
        .agg(
            discharge_occurrence_probability=(prob_col, "mean"),
            discharge_occurrence_count=("is_discharge", "sum"),
            n_bins=("is_discharge", "count"),
        )
    )
    B_tbl["discharge_occurrence_rate"] = B_tbl["discharge_occurrence_count"] / B_tbl["n_bins"]

    # Onset frequency: earliest threshold-crossing channels per seizure.
    all_channels = sorted(early["clean_channel_name"].unique(), key=natural_channel_key)
    onset_counts = {ch: 0 for ch in all_channels}
    onset_events = []

    for sid in ["S1", "S2", "S3"]:
        d = early[early["seizure_id"] == sid].copy()
        if d.empty:
            continue

        pivot = d.pivot_table(
            index="target_rel_start_sec",
            columns="clean_channel_name",
            values=prob_col,
            aggfunc="mean"
        ).sort_index()

        times = pivot.index.values
        chs = pivot.columns.tolist()
        mat = pivot.values.astype(float)

        hit = mat >= args.threshold

        if not hit.any():
            onset_events.append({
                "seizure_id": sid,
                "earliest_time_sec": np.nan,
                "onset_channels": "",
                "note": "no channel crossed threshold"
            })
            continue

        earliest_idx = np.where(hit.any(axis=1))[0][0]
        earliest_time = times[earliest_idx]
        onset_chs = [chs[j] for j in np.where(hit[earliest_idx])[0]]

        for ch in onset_chs:
            onset_counts[ch] += 1

        onset_events.append({
            "seizure_id": sid,
            "earliest_time_sec": float(earliest_time),
            "onset_channels": ";".join(onset_chs),
            "note": "model-derived earliest threshold crossing"
        })

    C_tbl = pd.DataFrame({
        "clean_channel_name": list(onset_counts.keys()),
        "model_derived_onset_frequency": list(onset_counts.values())
    })

    clinical_freq = (
        early.groupby(["seizure_id", "clean_channel_name"], as_index=False)["label"]
        .max()
        .groupby("clean_channel_name", as_index=False)["label"]
        .sum()
        .rename(columns={"label": "clinical_report_label_frequency"})
    )

    mean_prob = (
        A_tbl.groupby("clean_channel_name", as_index=False)["mean_predicted_probability"]
        .mean()
        .rename(columns={"mean_predicted_probability": "mean_prob_all_seizures"})
    )

    ranking = (
        mean_prob
        .merge(B_tbl, on="clean_channel_name", how="left")
        .merge(C_tbl, on="clean_channel_name", how="left")
        .merge(clinical_freq, on="clean_channel_name", how="left")
    )

    ranking = ranking.loc[:, ~ranking.columns.duplicated()].copy()
    ranking["model_derived_onset_frequency"] = ranking["model_derived_onset_frequency"].fillna(0)
    ranking["clinical_report_label_frequency"] = ranking["clinical_report_label_frequency"].fillna(0)

    ranking = ranking.sort_values(
        [
            "model_derived_onset_frequency",
            "clinical_report_label_frequency",
            "discharge_occurrence_probability",
            "mean_prob_all_seizures",
        ],
        ascending=[False, False, False, False]
    ).head(args.top_n)

    top_channels = ranking["clean_channel_name"].tolist()

    # Left panel: channel � time heatmap across all seizures.
    left_rows = []
    left_x_labels = []
    x_positions = []
    block_edges = []

    for sid in ["S1", "S2", "S3"]:
        d = early[early["seizure_id"] == sid].copy()
        pivot = d.pivot_table(
            index="clean_channel_name",
            columns="target_rel_start_sec",
            values=prob_col,
            aggfunc="mean"
        )

        pivot = pivot.reindex(index=top_channels)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        left_rows.append(pivot.values)
        start = len(left_x_labels)
        for t in pivot.columns:
            left_x_labels.append(f"{sid}\n{int(t)}s")
        end = len(left_x_labels)
        block_edges.append((start, end, sid))

    left_mat = np.concatenate(left_rows, axis=1)

    # Middle grid values.
    occurrence_map = dict(zip(B_tbl["clean_channel_name"], B_tbl["discharge_occurrence_probability"]))
    onset_map = dict(zip(C_tbl["clean_channel_name"], C_tbl["model_derived_onset_frequency"]))
    clinical_map = dict(zip(clinical_freq["clean_channel_name"], clinical_freq["clinical_report_label_frequency"]))

    grid_channels = top_channels
    occurrence_values = [occurrence_map.get(ch, 0) for ch in grid_channels]
    onset_values = [onset_map.get(ch, 0) for ch in grid_channels]
    clinical_positive_channels = [ch for ch in grid_channels if clinical_map.get(ch, 0) > 0]

    occ_mat, occ_labels, occ_rows, occ_cols = make_channel_grid(grid_channels, occurrence_values, "Epileptic discharge occurrence", "Occurrence", clinical_positive_channels, vmin=0, vmax=1)
    onset_mat, onset_labels, onset_rows, onset_cols = make_channel_grid(grid_channels, onset_values, "Onset channel frequency", "Frequency", clinical_positive_channels, vmin=0, vmax=3)

    # Plot.
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "savefig.dpi": 600,
    })

    fig = plt.figure(figsize=(18, 9), constrained_layout=True)
    gs = gridspec.GridSpec(
        2, 3,
        figure=fig,
        width_ratios=[2.4, 1.45, 1.0],
        height_ratios=[1, 1],
        wspace=0.18,
        hspace=0.18
    )

    ax_left = fig.add_subplot(gs[:, 0])
    ax_occ = fig.add_subplot(gs[0, 1])
    ax_onset = fig.add_subplot(gs[1, 1])
    ax_text = fig.add_subplot(gs[:, 2])

    cmap = "viridis"

    im_left = ax_left.imshow(left_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
    ax_left.set_title("Channel-level predicted probability", fontweight="bold")
    ax_left.set_ylabel("Channels")
    ax_left.set_xlabel("Multiple epileptic seizures")

    ax_left.set_yticks(range(len(top_channels)))
    ax_left.set_yticklabels(top_channels, fontsize=6)

    tick_positions = []
    tick_labels = []
    for start, end, sid in block_edges:
        mid = (start + end - 1) / 2
        tick_positions.append(mid)
        tick_labels.append(sid)
        if start > 0:
            ax_left.axvline(start - 0.5, color="white", linewidth=1.5)

    ax_left.set_xticks(tick_positions)
    ax_left.set_xticklabels(tick_labels)

    cb_left = fig.colorbar(im_left, ax=ax_left, fraction=0.025, pad=0.01)
    cb_left.set_label("Probability")

    # Occurrence grid.
    im_occ = ax_occ.imshow(occ_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=1, cmap=cmap)
    ax_occ.set_title("Epileptic discharge occurrence", fontweight="bold")
    ax_occ.set_xticks([])
    ax_occ.set_yticks([])

    for r in range(occ_rows):
        for c in range(occ_cols):
            if occ_labels[r][c]:
                ch = occ_labels[r][c]
                val = occ_mat[r, c]
                ax_occ.text(c, r, ch, ha="center", va="center", fontsize=6, color="white" if val < 0.55 else "black")
                if ch in clinical_positive_channels:
                    ax_occ.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=0.8))

    cb_occ = fig.colorbar(im_occ, ax=ax_occ, fraction=0.035, pad=0.01)
    cb_occ.set_label("Probability")

    # Onset frequency grid.
    vmax_onset = max(1, int(np.nanmax(onset_mat)))
    im_onset = ax_onset.imshow(onset_mat, aspect="auto", interpolation="nearest", vmin=0, vmax=vmax_onset, cmap=cmap)
    ax_onset.set_title("Onset channel frequency", fontweight="bold")
    ax_onset.set_xticks([])
    ax_onset.set_yticks([])

    for r in range(onset_rows):
        for c in range(onset_cols):
            if onset_labels[r][c]:
                ch = onset_labels[r][c]
                val = onset_mat[r, c]
                ax_onset.text(c, r, f"{ch}\n{int(val)}", ha="center", va="center", fontsize=6, color="white" if val < 0.6 * vmax_onset else "black")
                if ch in clinical_positive_channels:
                    ax_onset.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, edgecolor="red", linewidth=0.8))

    cb_onset = fig.colorbar(im_onset, ax=ax_onset, fraction=0.035, pad=0.01)
    cb_onset.set_label("Times as onset site")

    # Right explanatory panel instead of fake brain coordinates.
    ax_text.axis("off")
    top10 = ranking.head(10).copy()

    report_text = [
        "Candidate SOZ / onset contacts",
        "",
        f"Threshold = {args.threshold}",
        f"Window = {args.analysis_start:.0f} to +{args.analysis_end:.0f} sec",
        "",
        "Top ranked channels:"
    ]

    for i, row in enumerate(top10.itertuples(), start=1):
        report_text.append(
            f"{i}. {row.clean_channel_name}  "
            f"freq={int(row.model_derived_onset_frequency)}, "
            f"prob={row.mean_prob_all_seizures:.3f}"
        )

    report_text += [
        "",
        "Red boxes = report-derived",
        "earliest ictal-change contacts.",
        "",
        "True 3D brain pinpointing requires",
        "electrode coordinates from CT/MRI."
    ]

    ax_text.text(
        0.02, 0.98,
        "\n".join(report_text),
        va="top",
        ha="left",
        fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="black", linewidth=0.8)
    )

    # Panel labels.
    ax_left.text(-0.08, 1.02, "a", transform=ax_left.transAxes, fontsize=16, fontweight="bold")
    ax_occ.text(-0.10, 1.04, "b", transform=ax_occ.transAxes, fontsize=16, fontweight="bold")
    ax_onset.text(-0.10, 1.04, "c", transform=ax_onset.transAxes, fontsize=16, fontweight="bold")

    # Save tables.
    A_tbl.to_csv(out_dir / "paper_style_A_channel_level_predicted_probability.csv", index=False)
    B_tbl.to_csv(out_dir / "paper_style_B_epileptic_discharge_occurrence.csv", index=False)
    C_tbl.to_csv(out_dir / "paper_style_C_model_derived_onset_channel_frequency.csv", index=False)
    pd.DataFrame(onset_events).to_csv(out_dir / "paper_style_C_model_derived_onset_events_by_seizure.csv", index=False)
    ranking.to_csv(out_dir / "paper_style_brainwave_channel_ranking.csv", index=False)

    base = out_dir / "fig_brainwave_paper_style_liuwenqu_S1S2S3"
    fig.savefig(str(base) + ".png", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".pdf", dpi=600, bbox_inches="tight")
    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")
    plt.close(fig)

    print("\nSaved:")
    print(str(base) + ".png")
    print(str(base) + ".pdf")
    print(str(base) + ".tiff")

    print("\nTop ranked channels:")
    print(ranking.head(20).to_string(index=False))

    print("\nOnset events:")
    print(pd.DataFrame(onset_events).to_string(index=False))


if __name__ == "__main__":
    main()
