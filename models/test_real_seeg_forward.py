import torch
import mne
import numpy as np

from epibrant500m import EpiBRANT500M



# ======================================================
# Select real SEEG file
# ======================================================

EDF_FILE = (
"/home/ubuntu/Ijaz/SEEG datast/"
"pretrainingdataset/newSEEG data/63/1.edf"
)



DEVICE = "cuda"



# ======================================================
# Load EDF
# ======================================================

raw = mne.io.read_raw_edf(
    EDF_FILE,
    preload=True,
    verbose=False
)


fs = int(
    raw.info["sfreq"]
)


print("====================")
print("EDF INFORMATION")
print("====================")

print(
    "Sampling rate:",
    fs
)

print(
    "Channels:",
    len(raw.ch_names)
)

print(
    "Duration:",
    raw.times[-1]/60,
    "minutes"
)



# ======================================================
# Select SEEG channels
# remove auxiliary channels
# ======================================================


bad_keywords = [

    "ECG",
    "EKG",
    "EMG",
    "EOG",
    "RESP",
    "SPO2",
    "STATUS"

]


keep=[]


for ch in raw.ch_names:

    if not any(
        k in ch.upper()
        for k in bad_keywords
    ):
        keep.append(ch)



raw.pick(
    keep
)



print(
    "SEEG channels:",
    len(raw.ch_names)
)



# ======================================================
# Extract 60 seconds
# ======================================================


duration = 60


samples = int(
    fs*duration
)



x = raw.get_data(
    start=0,
    stop=samples
)



# x:

# channels x samples


x = torch.tensor(
    x,
    dtype=torch.float32
)



x = x.unsqueeze(0)



# B,C,T

print(
    "Input:",
    x.shape
)



x=x.to(
    DEVICE
)



# ======================================================
# Build model
# ======================================================


model = EpiBRANT500M(

    d_model=1024,

    temporal_layers=24,

    channel_layers=8,

    heads=16,

    max_channels=288

)


model=model.to(
    DEVICE
)



model.eval()



# ======================================================
# Forward
# ======================================================


with torch.no_grad():


    output = model(

        x,

        fs

    )



print("====================")
print("MODEL OUTPUT")
print("====================")


for k,v in output.items():

    if torch.is_tensor(v):

        print(
            k,
            v.shape
        )



print("SUCCESS")