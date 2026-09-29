import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from downstream.soz_localization.soz_dataset import (
    SOZLocalizationDataset
)

from downstream.soz_localization.soz_model import (
    EpiBRANTSOZLocalizer
)


# ======================================================
# PATHS
# ======================================================


MANIFEST = (
    "downstream/soz_localization/manifests/soz_manifest.csv"
)


CHECKPOINT = (
"/home/ubuntu/Ijaz/Revised code /BrainWae model "
"/SEEG_EpiBRANT500M/runs/"
"EpiBRANT500M_pilot500/checkpoints/"
"final_pretrained.pt"
)


OUTPUT_DIR = Path(
    "downstream/soz_localization/embeddings"
)


OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)



# ======================================================
# DEVICE
# ======================================================

device=torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("DEVICE:",device)



# ======================================================
# DATASET
# ======================================================

dataset = SOZLocalizationDataset(
    MANIFEST
)



loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)



# ======================================================
# MODEL
# ======================================================


model = EpiBRANTSOZLocalizer(
    CHECKPOINT,
    freeze_encoder=True
)


model.to(device)

model.eval()



# ======================================================
# STORAGE
# ======================================================


patient_storage={}



# ======================================================
# EXTRACTION
# ======================================================


with torch.no_grad():


    for idx,batch in enumerate(loader):


        signal=batch["signal"].to(
            device,
            non_blocking=True
        )


        mask=batch["channel_mask"].to(
            device,
            non_blocking=True
        )


        fs=batch["fs"].to(
            device,
            non_blocking=True
        )


        patient=batch["patient_id"][0]



        embedding=model.encode_contacts(
            signal,
            fs,
            mask
        )


        embedding=embedding.cpu()



        if patient not in patient_storage:

            patient_storage[patient]={
                
                "embedding":[],
                "soz_label":
                    batch["soz_label"][0].cpu(),

                "channel_mask":
                    batch["channel_mask"][0].cpu()
            }



        patient_storage[patient]["embedding"].append(
            embedding.squeeze(0)
        )



        if idx%50==0:

            print(
                f"Processed {idx}/{len(dataset)}"
            )



# ======================================================
# SAVE
# ======================================================


print("\nSaving embeddings")


for patient,data in patient_storage.items():


    embeddings=torch.stack(
        data["embedding"]
    )


    save_file = OUTPUT_DIR / f"{patient}.pt"



    torch.save(

        {

        "embedding":
            embeddings,


        "soz_label":
            data["soz_label"],


        "channel_mask":
            data["channel_mask"],


        "patient_id":
            patient

        },

        save_file

    )


    print(
        patient,
        embeddings.shape
    )



print("\nDONE")

print(
"Saved:",
len(patient_storage),
"patients"
)
