import pandas as pd
import numpy as np

from pathlib import Path

import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)



# =====================================================
# PATH
# =====================================================


RESULT_DIR = Path(
    "downstream/soz_localization/evaluation/results"
)


FIG_DIR = RESULT_DIR / "figures"

FIG_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# =====================================================
# Load patient results
# =====================================================


df = pd.read_csv(
    RESULT_DIR /
    "patient_metrics.csv"
)


print(df)



# =====================================================
# Figure 1
# Patient AUROC
# =====================================================


plt.figure(
    figsize=(8,4),
    dpi=600
)


plt.bar(
    range(len(df)),
    df["AUROC"]
)


plt.xticks(
    range(len(df)),
    df["Patient"],
    rotation=90,
    fontsize=6
)


plt.ylabel(
    "AUROC"
)


plt.xlabel(
    "Patients"
)


plt.title(
    "Patient-level SOZ localization AUROC"
)


plt.tight_layout()


plt.savefig(
    FIG_DIR /
    "Figure_S6A_patient_AUROC.png",
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    FIG_DIR /
    "Figure_S6A_patient_AUROC.pdf",
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Figure 2
# Dice distribution
# =====================================================


plt.figure(
    figsize=(4,5),
    dpi=600
)


plt.boxplot(
    df["Dice"],
    showmeans=True
)


plt.ylabel(
    "Dice coefficient"
)


plt.title(
    "SOZ localization overlap"
)


plt.tight_layout()


plt.savefig(
    FIG_DIR /
    "Figure_S6B_Dice_distribution.png",
    dpi=600,
    bbox_inches="tight"
)


plt.savefig(
    FIG_DIR /
    "Figure_S6B_Dice_distribution.pdf",
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Figure 3
# AUPRC distribution
# =====================================================


plt.figure(
    figsize=(8,4),
    dpi=600
)


plt.bar(
    range(len(df)),
    df["AUPRC"]
)


plt.xticks(
    range(len(df)),
    df["Patient"],
    rotation=90,
    fontsize=6
)


plt.ylabel(
    "AUPRC"
)


plt.title(
    "Patient-level precision-recall performance"
)


plt.tight_layout()


plt.savefig(
    FIG_DIR /
    "Figure_S6C_patient_AUPRC.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Figure 4
# Top-K localization
# =====================================================


def precision_recall_at_k(row,k):

    # approximate clinical ranking metric
    # based on available patient statistics

    soz=row.SOZ_contacts
    channels=row.Channels

    return min(
        k,
        soz
    ) / k



ks=[1,3,5,10]


topk=[]


for k in ks:

    values=[]

    for _,r in df.iterrows():

        values.append(
            precision_recall_at_k(
                r,
                k
            )
        )


    topk.append(
        np.mean(values)
    )



plt.figure(
    figsize=(5,4),
    dpi=600
)


plt.plot(
    ks,
    topk,
    marker="o"
)


plt.xlabel(
    "Top-K predicted contacts"
)


plt.ylabel(
    "Localization precision"
)


plt.title(
    "Top-K SOZ contact localization"
)


plt.xticks(
    ks
)


plt.tight_layout()


plt.savefig(
    FIG_DIR /
    "Figure_S6D_TopK_localization.png",
    dpi=600,
    bbox_inches="tight"
)


plt.close()



# =====================================================
# Summary
# =====================================================


summary=pd.DataFrame(

{
"Metric":
[
"Mean AUROC",
"Mean AUPRC",
"Mean Dice"
],

"Value":
[
df.AUROC.mean(),
df.AUPRC.mean(),
df.Dice.mean()
]

}

)


summary.to_csv(
RESULT_DIR /
"summary_metrics.csv",
index=False
)



print("\nSaved figures:")
for x in FIG_DIR.iterdir():
    print(x)


print("\nSummary:")
print(summary)
