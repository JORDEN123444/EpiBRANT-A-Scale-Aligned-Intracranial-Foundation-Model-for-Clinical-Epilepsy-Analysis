import torch

from torch.utils.data import DataLoader

from downstream.soz_localization.soz_dataset import (
    SOZLocalizationDataset
)

from downstream.seizure_detection.seizure_model import (
    EpiBRANTSeizureClassifier
)



MANIFEST = (
"downstream/soz_localization/manifests/soz_manifest.csv"
)


CHECKPOINT = (
"/home/ubuntu/Ijaz/Revised code /BrainWae model "
"/SEEG_EpiBRANT500M/runs/"
"EpiBRANT500M_pilot500/checkpoints/"
"final_pretrained.pt"
)



device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


print("DEVICE:",device)



dataset = SOZLocalizationDataset(
    MANIFEST
)


loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False
)


batch = next(iter(loader))


signal = batch["signal"].to(device)

mask = batch["channel_mask"].to(device)

fs = batch["fs"].to(device)



print("\nINPUT")
print("----------------")
print("Signal:",signal.shape)
print("Mask:",mask.shape)
print("FS:",fs)



model = EpiBRANTSeizureClassifier(
    CHECKPOINT
)


model.to(device)

model.eval()



with torch.no_grad():

    embedding = model.encode(
        signal,
        fs,
        mask
    )


print("\nOUTPUT")
print("----------------")

print(
"Embedding:",
embedding.shape
)


print(
"Mean:",
embedding.mean().item()
)


print("\nFORWARD TEST PASSED")

