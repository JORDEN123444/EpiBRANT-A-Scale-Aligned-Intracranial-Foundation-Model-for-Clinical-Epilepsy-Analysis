from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.patches import Rectangle


# =========================================================
# PATHS
# =========================================================

ROOT = Path("downstream/soz_localization/evaluation/results")
FIG_DIR = ROOT / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PATIENT_METRICS = ROOT / "patient_metrics.csv"
CONTACT_PRED = ROOT / "contact_predictions.csv"
CONTACT_SUMMARY = ROOT / "contact_prediction_summary.csv"


# =========================================================
# STYLE (Nature-like clean format)
# =========================================================

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.linewidth": 0.8,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
})


# =========================================================
# LOAD
# =========================================================

pm = pd.read_csv(PATIENT_METRICS)
cp = pd.read_csv(CONTACT_PRED)
cs = pd.read_csv(CONTACT_SUMMARY)

# choose 4 representative patients: top 4 by Dice
top4 = pm.sort_values("Dice", ascending=False).head(4)["Patient"].tolist()

# colors for panel a
point_colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52"]


def short_name(x):
    x = x.replace("sub-", "")
    x = x.replace("openieeg", "")
    return x


# =========================================================
# PANEL A DATA
# =========================================================

a_df = pm.merge(cs, left_on="Patient", right_on="patient_id", how="left")
a_df = a_df[a_df["Patient"].isin(top4)].copy()
a_df["short"] = a_df["Patient"].apply(short_name)

# sort by Dice descending to keep best examples
a_df = a_df.sort_values("Dice", ascending=False).reset_index(drop=True)


# =========================================================
# PANEL B/C/D MATRICES
# Sorted by predicted probability, one column per patient
# =========================================================

selected = a_df["Patient"].tolist()
subsets = []

max_contacts = 0
for pid in selected:
    temp = cp[cp["patient_id"] == pid].sort_values("probability", ascending=False).reset_index(drop=True)
    subsets.append(temp)
    max_contacts = max(max_contacts, len(temp))

prob_mat = np.full((max_contacts, len(selected)), np.nan)
truth_mat = np.full((max_contacts, len(selected)), np.nan)
pred_mat = np.full((max_contacts, len(selected)), np.nan)

top_contact_names = {}

for j, pid in enumerate(selected):
    temp = subsets[j]
    n = len(temp)
    prob_mat[:n, j] = temp["probability"].values
    truth_mat[:n, j] = temp["true_soz"].values
    pred_mat[:n, j] = temp["pred_label"].values

    top_contact_names[pid] = temp.head(8)[["contact_name", "probability", "true_soz", "anatomical"]]


# =========================================================
# FIGURE LAYOUT
# =========================================================

fig = plt.figure(figsize=(13.5, 5.8))
outer = gridspec.GridSpec(
    1, 3,
    width_ratios=[1.1, 1.8, 1.7],
    wspace=0.30
)

# left nested panel a
left = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=outer[0],
    hspace=0.28
)

# middle big heatmap b
mid = gridspec.GridSpecFromSubplotSpec(
    1, 1,
    subplot_spec=outer[1]
)

# right stacked c/d
right = gridspec.GridSpecFromSubplotSpec(
    2, 1,
    subplot_spec=outer[2],
    hspace=0.35,
    height_ratios=[1.0, 1.15]
)

ax_a1 = fig.add_subplot(left[0])
ax_a2 = fig.add_subplot(left[1])
ax_b = fig.add_subplot(mid[0])
ax_c = fig.add_subplot(right[0])
ax_d = fig.add_subplot(right[1])


# =========================================================
# PANEL A1: Sensitivity vs Specificity
# =========================================================

for i, row in a_df.iterrows():
    ax_a1.scatter(
        row["sensitivity"],
        row["specificity"],
        s=55,
        color=point_colors[i],
        edgecolor="black",
        linewidth=0.4,
        label=row["short"]
    )

ax_a1.set_xlabel("Sensitivity")
ax_a1.set_ylabel("Specificity")
ax_a1.set_xlim(0, 1.02)
ax_a1.set_ylim(0, 1.02)
ax_a1.set_title("a", loc="left", fontweight="bold")
ax_a1.legend(
    frameon=False,
    loc="upper left",
    borderpad=0.2,
    handletextpad=0.3
)

# =========================================================
# PANEL A2: AUROC vs AUPRC
# =========================================================

for i, row in a_df.iterrows():
    ax_a2.scatter(
        row["AUROC"],
        row["AUPRC"],
        s=55,
        color=point_colors[i],
        edgecolor="black",
        linewidth=0.4
    )

ax_a2.set_xlabel("AUROC")
ax_a2.set_ylabel("AP")
ax_a2.set_xlim(0, 1.02)
ax_a2.set_ylim(0, 1.02)


# =========================================================
# PANEL B: Channel-level predicted probability
# =========================================================

im_b = ax_b.imshow(
    prob_mat,
    aspect="auto",
    cmap="viridis",
    vmin=0.0,
    vmax=1.0,
    interpolation="nearest"
)

ax_b.set_title("b", loc="left", fontweight="bold")
ax_b.set_title("Channel-level predicted probability", fontsize=10)
ax_b.set_xlabel("Patients")
ax_b.set_ylabel("Ranked contacts")
ax_b.set_xticks(np.arange(len(selected)))
ax_b.set_xticklabels([short_name(x) for x in selected], rotation=0)
ax_b.set_yticks([])

cbar_b = fig.colorbar(im_b, ax=ax_b, fraction=0.03, pad=0.02)
cbar_b.set_label("Probability", rotation=90)


# =========================================================
# PANEL C: Clinical SOZ annotation sorted by predicted probability
# =========================================================

im_c = ax_c.imshow(
    truth_mat,
    aspect="auto",
    cmap="cividis",
    vmin=0,
    vmax=1,
    interpolation="nearest"
)

ax_c.set_title("c", loc="left", fontweight="bold")
ax_c.set_title("Clinical SOZ annotation", fontsize=10)
ax_c.set_xlabel("Patients")
ax_c.set_ylabel("Ranked contacts")
ax_c.set_xticks(np.arange(len(selected)))
ax_c.set_xticklabels([short_name(x) for x in selected], rotation=0)
ax_c.set_yticks([])

# red rectangle around top 10 ranked contacts
rect1 = Rectangle(
    (-0.5, -0.5),
    width=len(selected),
    height=min(10, max_contacts),
    fill=False,
    edgecolor="red",
    linewidth=1.5,
    linestyle="--"
)
ax_c.add_patch(rect1)
ax_c.text(
    len(selected) - 0.2,
    min(10, max_contacts) + 1.5,
    "Top-ranked\ncontacts",
    color="red",
    ha="right",
    va="bottom",
    fontsize=8
)


# =========================================================
# PANEL D: Top-contact localization summary table-like plot
# =========================================================

ax_d.axis("off")
ax_d.set_title("d", loc="left", fontweight="bold")
ax_d.set_title("Top localized contacts", fontsize=10)

y0 = 0.95
dy = 0.21

for idx, pid in enumerate(selected):
    temp = top_contact_names[pid]
    header = f"{short_name(pid)}"
    ax_d.text(
        0.01,
        y0 - idx * dy,
        header,
        fontweight="bold",
        fontsize=8,
        va="top"
    )

    lines = []
    for _, r in temp.head(4).iterrows():
        soz_mark = "*" if int(r["true_soz"]) == 1 else ""
        anat = "" if pd.isna(r["anatomical"]) else str(r["anatomical"])
        line = f"{r['contact_name']}  ({r['probability']:.2f}) {soz_mark}"
        if anat not in ["", "-1", "nan"]:
            line += f"  [{anat}]"
        lines.append(line)

    ax_d.text(
        0.18,
        y0 - idx * dy,
        "\n".join(lines),
        fontsize=7.2,
        va="top",
        family="monospace"
    )


# =========================================================
# FINAL SAVE
# =========================================================

plt.tight_layout()

png_path = FIG_DIR / "Figure_S7_SOZ_localization_nature.png"
pdf_path = FIG_DIR / "Figure_S7_SOZ_localization_nature.pdf"
tiff_path = FIG_DIR / "Figure_S7_SOZ_localization_nature.tiff"

plt.savefig(png_path, dpi=600, bbox_inches="tight")
plt.savefig(pdf_path, bbox_inches="tight")
plt.savefig(tiff_path, dpi=600, bbox_inches="tight")

print("Saved:")
print(png_path)
print(pdf_path)
print(tiff_path)

