#!/usr/bin/env python3

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


# ==========================================================
# PATHS
# ==========================================================

MANIFEST = "global_pretrain_manifest.csv"

OUT_DIR = Path("dataset_QC_plots")

OUT_DIR.mkdir(
    exist_ok=True
)



# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(MANIFEST)



# ==========================================================
# Rename datasets
# ==========================================================

df["dataset"] = df["dataset"].replace({

    "Institutional_SEEG":
        "Private-SEEG",

    "Omni_iEEG_SEEG":
        "Omni-iEEG"

})



# ==========================================================
# NPJ STYLE
# ==========================================================

sns.set_theme(
    style="white"
)


plt.rcParams.update({

    "font.family": "Arial",

    "font.size": 8,

    "axes.linewidth": 0.8,

    "axes.labelsize": 9,

    "xtick.labelsize": 8,

    "ytick.labelsize": 8

})



# Dataset colors

dataset_colors = {

    "Private-SEEG":
        "#1f77b4",

    "Omni-iEEG":
        "#ff7f0e"

}



# ==========================================================
# A. Sampling frequency distribution
# ==========================================================


fs = (

    df["sampling_rate"]
    .round()
    .value_counts()
    .sort_index()

)



plt.figure(
    figsize=(3.2,2.6),
    dpi=600
)



ax = sns.barplot(

    x=fs.index.astype(str),

    y=fs.values,

    color="#1f77b4"

)



plt.xlabel(
    "Sampling frequency (Hz)"
)


plt.ylabel(
    "Number of recordings"
)


sns.despine()


plt.tight_layout()



plt.savefig(

    OUT_DIR /
    "A_sampling_frequency_distribution.pdf",

    bbox_inches="tight"

)


plt.savefig(

    OUT_DIR /
    "A_sampling_frequency_distribution.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



# ==========================================================
# B. SEEG channel distribution by dataset
# ==========================================================


plt.figure(
    figsize=(3.3,2.8),
    dpi=600
)



sns.histplot(

    data=df,

    x="seeg_channels",

    hue="dataset",

    bins=25,

    kde=True,

    palette=dataset_colors,

    alpha=0.45

)



plt.xlabel(
    "Number of SEEG contacts"
)


plt.ylabel(
    "Number of recordings"
)



plt.legend(

    title="",

    frameon=False

)



sns.despine()


plt.tight_layout()



plt.savefig(

    OUT_DIR /
    "B_SEEG_channel_distribution.pdf",

    bbox_inches="tight"

)


plt.savefig(

    OUT_DIR /
    "B_SEEG_channel_distribution.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



# ==========================================================
# C. Recording duration distribution
# ==========================================================


plt.figure(

    figsize=(3.2,2.6),

    dpi=600

)



sns.histplot(

    data=df,

    x="duration_hours",

    hue="dataset",

    bins=30,

    palette=dataset_colors,

    alpha=0.5

)



plt.xlabel(
    "Recording duration (hours)"
)


plt.ylabel(
    "Number of recordings"
)



plt.legend(

    title="",

    frameon=False

)



sns.despine()


plt.tight_layout()



plt.savefig(

    OUT_DIR /
    "C_recording_duration_distribution.pdf",

    bbox_inches="tight"

)


plt.savefig(

    OUT_DIR /
    "C_recording_duration_distribution.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



# ==========================================================
# D. Channel variability comparison
# ==========================================================


plt.figure(

    figsize=(3.2,2.8),

    dpi=600

)



sns.boxplot(

    data=df,

    x="dataset",

    y="seeg_channels",

    palette=dataset_colors

)



plt.xlabel("")


plt.ylabel(
    "SEEG contacts"
)



plt.xticks(

    rotation=25,

    ha="right"

)



sns.despine()


plt.tight_layout()



plt.savefig(

    OUT_DIR /
    "D_dataset_channel_comparison.pdf",

    bbox_inches="tight"

)


plt.savefig(

    OUT_DIR /
    "D_dataset_channel_comparison.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print("==============================")
print("QC plots generated")
print("==============================")

print(
    OUT_DIR
)