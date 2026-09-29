import torch

from downstream.soz_localization.soz_dataset import (
    SOZLocalizationDataset
)

from downstream.soz_localization.soz_model import (
    EpiBRANTSOZLocalizer
)


MANIFEST="downstream/soz_localization/manifests/soz_manifest.csv"


CHECKPOINT="/home/ubuntu/Ijaz/Revised code /BrainWae model /SEEG_EpiBRANT500M/runs/EpiBRANT500M_pilot500/checkpoints/final_pretrained.pt"



device="cuda"



dataset=SOZLocalizationDataset(
    MANIFEST
)


sample=dataset[0]


signal=sample["signal"].unsqueeze(0).to(device)

mask=sample["channel_mask"].unsqueeze(0).to(device)

fs=sample["fs"].unsqueeze(0).to(device)



model=EpiBRANTSOZLocalizer(
    CHECKPOINT
)


model.to(device)

model.eval()



with torch.no_grad():

    emb=model.encode_contacts(
        signal,
        fs,
        mask
    )


    logits=model(
        signal,
        fs,
        mask
    )



print("==============================")

print(
"Embedding:",
emb.shape
)


print(
"SOZ logits:",
logits.shape
)


print("==============================")

