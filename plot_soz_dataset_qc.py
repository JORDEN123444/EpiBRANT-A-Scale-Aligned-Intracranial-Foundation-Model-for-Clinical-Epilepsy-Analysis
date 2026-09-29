from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# =====================================================
# PATHS
# =====================================================

manifest = Path(
    "downstream/soz_localization/manifests/soz_manifest.csv"
)


outdir = Path(
    "downstream/soz_localization/figures"
)

outdir.mkdir(
    parents=True,
    exist_ok=True
)



# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    manifest
)


patient_channels = (
    df.groupby("patient_id")
    ["num_seeg_channels"]
    .first()
    .sort_values()
)



# =====================================================
# FIGURE S1
# SEEG channel number per patient
# =====================================================


plt.figure(
    figsize=(10,4),
    dpi=300
)


plt.bar(
    patient_channels.index,
    patient_channels.values
)


plt.xticks(
    rotation=70,
    fontsize=7
)


plt.ylabel(
    "Number of SEEG contacts",
    fontsize=11
)


plt.xlabel(
    "Patient",
    fontsize=11
)


plt.title(
    "Distribution of SEEG contacts across patients",
    fontsize=12
)


plt.tight_layout()


plt.savefig(
    outdir /
    "Supplementary_Figure_S1_SEEG_channels.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# FIGURE S2
# Channel distribution histogram
# =====================================================


values = patient_channels.values



plt.figure(
    figsize=(5,4),
    dpi=300
)


plt.hist(
    values,
    bins=8
)


plt.xlabel(
    "Number of SEEG contacts"
)


plt.ylabel(
    "Number of patients"
)


plt.title(
    "SEEG channel distribution"
)


plt.tight_layout()


plt.savefig(
    outdir /
    "Supplementary_Figure_S2_channel_histogram.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# FIGURE S3
# SOZ CONTACT RATIO
# =====================================================


soz_ratio=[]


for patient,row in patient_channels.items():


    # get one channels file
    ch_file = (
        df[df.patient_id==patient]
        .channels_file
        .iloc[0]
    )


    channels=pd.read_csv(
        ch_file,
        sep="\t"
    )


    seeg = channels[
        channels.type
        .astype(str)
        .str.upper()
        ==
        "SEEG"
    ]


    total=len(seeg)


    soz = (
        seeg["soz"]
        .sum()
    )


    ratio = (
        soz / total * 100
    )


    soz_ratio.append(
        ratio
    )



soz_df=pd.DataFrame(
    {
        "patient":
        patient_channels.index,

        "soz_percentage":
        soz_ratio
    }
)


soz_df=soz_df.sort_values(
    "soz_percentage"
)



plt.figure(
    figsize=(10,4),
    dpi=300
)


plt.bar(
    soz_df.patient,
    soz_df.soz_percentage
)


plt.xticks(
    rotation=70,
    fontsize=7
)


plt.ylabel(
    "SOZ contacts (%)"
)


plt.xlabel(
    "Patient"
)


plt.title(
    "Clinical SOZ contact distribution"
)


plt.tight_layout()


plt.savefig(
    outdir /
    "Supplementary_Figure_S3_SOZ_distribution.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



print("="*70)

print("QC FIGURES GENERATED")

print(outdir)

print("="*70)


print("\nChannel statistics")

print(patient_channels.describe())


print("\nSOZ percentage")

print(
soz_df.soz_percentage.describe()
)
