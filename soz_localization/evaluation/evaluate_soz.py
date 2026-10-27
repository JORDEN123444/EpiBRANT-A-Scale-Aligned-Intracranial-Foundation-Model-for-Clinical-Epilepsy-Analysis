import os
import torch
import numpy as np
import pandas as pd

from pathlib import Path

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    precision_recall_curve
)


# ============================================================
# PATHS
# ============================================================


ROOT = Path(
    "downstream/soz_localization"
)


EMB_DIR = ROOT / "embeddings"


CKPT_DIR = (
    ROOT /
    "training" /
    "checkpoints"
)


OUT_DIR = (
    ROOT /
    "evaluation" /
    "results"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



DEVICE="cuda" if torch.cuda.is_available() else "cpu"



# ============================================================
# SOZ HEAD
# ============================================================


import torch.nn as nn


class SOZHead(nn.Module):

    def __init__(
        self,
        dim=1024
    ):

        super().__init__()

        self.norm = nn.LayerNorm(
            dim
        )

        self.fc = nn.Linear(
            dim,
            1
        )


    def forward(
        self,
        x
    ):

        x = self.norm(
            x
        )

        x = self.fc(
            x
        )

        return x.squeeze(-1)



# ============================================================
# LOAD MODEL
# ============================================================


def load_head(checkpoint):

    model = SOZHead()

    state = torch.load(
        checkpoint,
        map_location="cpu"
    )


    model.load_state_dict(
        state["model"]
    )


    model.eval()

    return model



# ============================================================
# METRICS
# ============================================================


def dice_score(
    pred,
    true
):

    intersection = (
        pred *
        true
    ).sum()


    return (
        2*intersection /
        (
            pred.sum()
            +
            true.sum()
            +
            1e-8
        )
    )



# ============================================================
# MAIN
# ============================================================


all_results=[]


roc_y=[]
roc_p=[]


patients = sorted(
    [
        x.stem
        for x in EMB_DIR.glob("*.pt")
    ]
)



print(
    "Patients:",
    len(patients)
)



for fold, patient in enumerate(patients,1):


    print(
        "\n=============================="
    )

    print(
        "Testing:",
        patient
    )


    # LOSO:
    # fold model corresponds to held-out patient

    ckpt = (
        CKPT_DIR /
        f"fold_{fold}.pt"
    )


    if not ckpt.exists():

        print(
            "Missing:",
            ckpt
        )

        continue



    model = load_head(
        ckpt
    )



    data=torch.load(
        EMB_DIR /
        f"{patient}.pt",
        map_location="cpu"
    )


    embedding=data["embedding"]

    label = data["soz_label"]

    mask = data["channel_mask"]



    # -------------------------------------------------
    # Average all windows
    # -------------------------------------------------


    embedding=embedding.mean(
        dim=0
    )


    valid=mask.bool()


    embedding=embedding[valid]

    label=label[valid]



    with torch.no_grad():

        logits=model(
            embedding
        )


        prob=torch.sigmoid(
            logits
        ).numpy()



    y=label.numpy()



    pred_binary=(
        prob>=0.5
    ).astype(int)



    # metrics


    auroc=roc_auc_score(
        y,
        prob
    )


    auprc=average_precision_score(
        y,
        prob
    )


    dice=dice_score(
        pred_binary,
        y
    )



    precision=precision_score(
        y,
        pred_binary,
        zero_division=0
    )


    recall=recall_score(
        y,
        pred_binary,
        zero_division=0
    )



    f1=f1_score(
        y,
        pred_binary,
        zero_division=0
    )



    print(
        patient,
        "AUROC",
        round(auroc,3),
        "Dice",
        round(float(dice),3)
    )



    all_results.append(
        [
            patient,
            len(y),
            int(y.sum()),
            auroc,
            auprc,
            dice,
            precision,
            recall,
            f1
        ]
    )



    roc_y.extend(
        y
    )

    roc_p.extend(
        prob
    )




# ============================================================
# SAVE TABLE
# ============================================================


columns=[

"Patient",
"Channels",
"SOZ_contacts",
"AUROC",
"AUPRC",
"Dice",
"Precision",
"Recall",
"F1"

]


df=pd.DataFrame(
    all_results,
    columns=columns
)



df.to_csv(
    OUT_DIR /
    "patient_metrics.csv",
    index=False
)



overall=pd.DataFrame(
    {

    "AUROC":[
        roc_auc_score(
            roc_y,
            roc_p
        )
    ],

    "AUPRC":[
        average_precision_score(
            roc_y,
            roc_p
        )
    ]

    }
)


overall.to_csv(
    OUT_DIR /
    "overall_metrics.csv",
    index=False
)



print("\nDONE")

print(df)
