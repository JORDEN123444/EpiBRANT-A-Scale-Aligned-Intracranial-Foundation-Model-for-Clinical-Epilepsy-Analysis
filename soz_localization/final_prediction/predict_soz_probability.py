# ============================================================
# EpiBRANT-500M
# Clinical SOZ Localization
# Contact-level probability prediction
# ============================================================


import os
import sys
import glob

import torch
import numpy as np
import pandas as pd



# ============================================================
# ADD PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../../"
    )
)

sys.path.insert(
    0,
    PROJECT_ROOT
)



# ============================================================
# IMPORT SOZ HEAD
# ============================================================

from downstream.soz_localization.training.soz_head import SOZHead



# ============================================================
# CONFIGURATION
# ============================================================

DEVICE = (
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print(
    "DEVICE:",
    DEVICE
)


EMBED_DIR = (
    "downstream/soz_localization/embeddings"
)


CHECKPOINT_DIR = (
    "downstream/soz_localization/training/checkpoints"
)


OUTPUT_DIR = (
    "downstream/soz_localization/final_prediction"
)


os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)



# ============================================================
# LOAD TRAINED SOZ MODEL
# ============================================================


def load_soz_model(checkpoint_path):

    print("\nLoading checkpoint:")
    print(checkpoint_path)


    model = SOZHead(
        dim=1024
    )


    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu"
    )


    state_dict = checkpoint["model"]


    model.load_state_dict(
        state_dict,
        strict=True
    )


    model.to(
        DEVICE
    )


    model.eval()


    print(
        "SOZHead loaded successfully"
    )


    print(
        "Checkpoint fold:",
        checkpoint["fold"]
    )


    print(
        "Test patient:",
        checkpoint["test_patient"]
    )


    return model



# ============================================================
# SELECT CHECKPOINT
# ============================================================


checkpoint = os.path.join(
    CHECKPOINT_DIR,
    "fold_1.pt"
)


soz_model = load_soz_model(
    checkpoint
)



# ============================================================
# LOAD ALL PATIENT EMBEDDINGS
# ============================================================


embedding_files = sorted(
    glob.glob(
        os.path.join(
            EMBED_DIR,
            "*.pt"
        )
    )
)


print(
    "\nPatients found:",
    len(embedding_files)
)



if len(embedding_files)==0:

    raise RuntimeError(
        "No embedding files found"
    )



# ============================================================
# PREDICTION
# ============================================================


summary=[]


for file in embedding_files:


    print(
        "\n================================"
    )


    data=torch.load(
        file,
        map_location="cpu"
    )


    patient=data[
        "patient_id"
    ]


    print(
        "Processing:",
        patient
    )


    embedding=data[
        "embedding"
    ]


    label=data[
        "soz_label"
    ]


    mask=data[
        "channel_mask"
    ]



    # --------------------------------------------------------
    # Temporal aggregation
    #
    # [windows, channels, 1024]
    #
    # becomes:
    #
    # [channels,1024]
    # --------------------------------------------------------


    contact_embedding = (
        embedding.mean(
            dim=0
        )
    )



    valid_channels = (
        mask.bool()
    )


    x = contact_embedding[
        valid_channels
    ]



    clinical_label = (
        label[
            valid_channels
        ]
        .numpy()
    )



    # --------------------------------------------------------
    # SOZ probability
    # --------------------------------------------------------

    with torch.no_grad():


        logits = soz_model(
            x.to(DEVICE)
        )


        probability = torch.sigmoid(
            logits
        )


    probability = (
        probability
        .cpu()
        .numpy()
        .reshape(-1)
    )



    # --------------------------------------------------------
    # SAVE CONTACT RESULTS
    # --------------------------------------------------------


    contacts=np.arange(
        len(probability)
    )


    patient_df=pd.DataFrame({

        "patient":
        patient,

        "contact_index":
        contacts,

        "SOZ_probability":
        probability,

        "clinical_SOZ":
        clinical_label

    })


    output_file=os.path.join(
        OUTPUT_DIR,
        patient +
        "_SOZ_probability.csv"
    )


    patient_df.to_csv(
        output_file,
        index=False
    )


    print(
        "Saved:",
        output_file
    )



    summary.append({

        "Patient":
        patient,

        "Channels":
        len(probability),

        "Clinical_SOZ_contacts":
        int(
            clinical_label.sum()
        ),

        "Maximum_probability":
        float(
            probability.max()
        ),

        "Mean_probability":
        float(
            probability.mean()
        )

    })



# ============================================================
# SAVE SUMMARY
# ============================================================


summary_df=pd.DataFrame(
    summary
)


summary_file=os.path.join(
    OUTPUT_DIR,
    "SOZ_probability_summary.csv"
)


summary_df.to_csv(
    summary_file,
    index=False
)



print("\n================================")
print("DONE")
print("================================")


print(summary_df)


print(
"\nSaved summary:"
)

print(
summary_file
)