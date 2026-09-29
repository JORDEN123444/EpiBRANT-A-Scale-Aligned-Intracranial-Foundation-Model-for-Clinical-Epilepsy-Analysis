from pathlib import Path

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from sklearn.decomposition import PCA
from sklearn.manifold import TSNE



# =====================================================
# PATHS
# =====================================================

EMB_DIR = Path(
    "downstream/soz_localization/embeddings"
)


OUT_DIR = Path(
    "downstream/soz_localization/figures"
)


OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# =====================================================
# LOAD EMBEDDINGS
# =====================================================


all_embeddings = []
all_labels = []

patient_windows = []
patient_soz = []



files = sorted(
    EMB_DIR.glob("*.pt")
)


print(
    "Patients found:",
    len(files)
)



for file in files:


    data = torch.load(
        file,
        map_location="cpu"
    )


    patient = data["patient_id"]

    embedding = data["embedding"]

    soz_label = data["soz_label"]

    mask = data["channel_mask"].bool()



    print(
        patient,
        "Embedding:",
        tuple(embedding.shape),
        "SOZ:",
        int(soz_label.sum())
    )



    # ------------------------------------
    # Patient statistics
    # ------------------------------------

    patient_windows.append(
        [
            patient,
            embedding.shape[0]
        ]
    )


    patient_soz.append(
        [
            patient,
            int(soz_label.sum()),
            int(mask.sum())
        ]
    )



    # ------------------------------------
    # Select real contacts
    # ------------------------------------


    embedding = embedding[:, mask, :]


    windows = embedding.shape[0]


    contacts = embedding.shape[1]


    # [windows, contacts,1024]
    #
    # ->
    #
    # [windows*contacts,1024]


    embedding = embedding.reshape(
        -1,
        1024
    )


    labels = soz_label[mask]


    labels = labels.repeat(
        windows
    )



    all_embeddings.append(
        embedding.numpy()
    )


    all_labels.append(
        labels.numpy()
    )




# =====================================================
# MERGE ALL PATIENTS
# =====================================================


X = np.concatenate(
    all_embeddings,
    axis=0
)


y = np.concatenate(
    all_labels
)



print("\n"+"="*70)

print(
    "Embedding matrix:",
    X.shape
)


print(
    "Total contact embeddings:",
    len(y)
)


print(
    "SOZ:",
    int(y.sum())
)


print(
    "Non-SOZ:",
    int((y==0).sum())
)

print("="*70)



# =====================================================
# Figure S4A
# Embedding magnitude distribution
# =====================================================


embedding_norm = np.linalg.norm(
    X,
    axis=1
)



plt.figure(
    figsize=(5,4),
    dpi=300
)


plt.hist(
    embedding_norm,
    bins=60
)


plt.xlabel(
    "Embedding L2 norm"
)


plt.ylabel(
    "Number of embeddings"
)


plt.title(
    "EpiBRANT contact embedding magnitude"
)



plt.tight_layout()


plt.savefig(

    OUT_DIR /
    "Supplementary_Figure_S4A_embedding_norm.png",

    dpi=600,

    bbox_inches="tight"

)


plt.close()



print(
    "Saved S4A"
)



# =====================================================
# Figure S4B
# PCA + t-SNE visualization
# =====================================================


print(
    "\nRunning PCA + t-SNE..."
)



# Limit points for visualization

max_points = 50000



if len(X) > max_points:


    rng = np.random.RandomState(
        42
    )


    idx = rng.choice(

        len(X),

        max_points,

        replace=False

    )


    X_vis = X[idx]

    y_vis = y[idx]


else:

    X_vis = X

    y_vis = y




# PCA

pca = PCA(
    n_components=50,
    random_state=42
)


X_pca = pca.fit_transform(
    X_vis
)



print(
    "PCA variance:",
    pca.explained_variance_ratio_.sum()
)



# t-SNE

tsne = TSNE(

    n_components=2,

    random_state=42,

    perplexity=30,

    max_iter=1000

)


Z = tsne.fit_transform(
    X_pca
)




plt.figure(

    figsize=(6,5),

    dpi=300

)



plt.scatter(

    Z[y_vis==0,0],

    Z[y_vis==0,1],

    s=5,

    alpha=0.3,

    label="Non-SOZ"

)



plt.scatter(

    Z[y_vis==1,0],

    Z[y_vis==1,1],

    s=8,

    alpha=0.7,

    label="SOZ"

)



plt.xlabel(
    "t-SNE dimension 1"
)


plt.ylabel(
    "t-SNE dimension 2"
)



plt.legend()



plt.title(
    "EpiBRANT contact embedding visualization"
)



plt.tight_layout()



plt.savefig(

    OUT_DIR /

    "Supplementary_Figure_S4B_embedding_TSNE.png",

    dpi=600,

    bbox_inches="tight"

)



plt.close()



print(
    "Saved S4B"
)



# =====================================================
# Figure S4C
# Windows per patient
# =====================================================


df_window = pd.DataFrame(

    patient_windows,

    columns=[

        "Patient",

        "Windows"

    ]

)



df_window = df_window.sort_values(
    "Windows"
)



plt.figure(

    figsize=(8,4),

    dpi=300

)



plt.bar(

    df_window.Patient,

    df_window.Windows

)



plt.xticks(

    rotation=75,

    fontsize=7

)



plt.ylabel(
    "Number of 60-second windows"
)


plt.xlabel(
    "Patient"
)


plt.title(
    "SEEG validation windows per patient"
)



plt.tight_layout()



plt.savefig(

    OUT_DIR /

    "Supplementary_Figure_S4C_patient_windows.png",

    dpi=600,

    bbox_inches="tight"

)



plt.close()



print(
    "Saved S4C"
)




# =====================================================
# Figure S4D
# SOZ distribution
# =====================================================


df_soz = pd.DataFrame(

    patient_soz,

    columns=[

        "Patient",

        "SOZ_contacts",

        "Total_contacts"

    ]

)



df_soz=df_soz.sort_values(
    "SOZ_contacts"
)




plt.figure(

    figsize=(8,4),

    dpi=300

)



plt.bar(

    df_soz.Patient,

    df_soz.SOZ_contacts

)



plt.xticks(

    rotation=75,

    fontsize=7

)



plt.ylabel(
    "SOZ contacts"
)


plt.xlabel(
    "Patient"
)



plt.title(
    "Clinical SOZ contact distribution"
)



plt.tight_layout()



plt.savefig(

    OUT_DIR /

    "Supplementary_Figure_S4D_SOZ_distribution.png",

    dpi=600,

    bbox_inches="tight"

)



plt.close()



print(
    "Saved S4D"
)




# =====================================================
# SAVE TABLE
# =====================================================


df_soz.to_csv(

    OUT_DIR /

    "SOZ_contact_summary.csv",

    index=False

)



print("\n================================")
print("ALL QC FIGURES COMPLETED")
print("Output:", OUT_DIR)
print("================================")