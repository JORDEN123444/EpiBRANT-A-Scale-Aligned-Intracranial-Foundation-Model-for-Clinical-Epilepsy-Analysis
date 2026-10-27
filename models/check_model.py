import torch
import torch.nn as nn

from epibrant500m import EpiBRANT500M



# ==========================================
# Device
# ==========================================

device = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", device)



# ==========================================
# Build model
# ==========================================


model = EpiBRANT500M(
    d_model=1024,
    temporal_layers=24,
    channel_layers=8,
    heads=16,
    max_channels=288
)


model.to(device)



# ==========================================
# Parameter count
# ==========================================


total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\n==========================")
print("MODEL PARAMETERS")
print("==========================")

print(
    "Total parameters:",
    total_params/1e6,
    "M"
)


print(
    "Trainable parameters:",
    trainable_params/1e6,
    "M"
)



# ==========================================
# Dummy SEEG input
# ==========================================

"""
Example:

batch = 1

channels = 128

sampling rate = 2000 Hz

duration = 60 sec

samples = 120000
"""


x = torch.randn(
    1,
    128,
    120000
).to(device)



channel_mask = torch.ones(
    1,
    128
).to(device)



sampling_rate = 2000



# ==========================================
# Forward test
# ==========================================


print("\n==========================")
print("FORWARD TEST")
print("==========================")


with torch.no_grad():

    output = model(
        x,
        sampling_rate,
        channel_mask
    )


print(
    "Forward successful"
)



print(
    "Output keys:",
    output.keys()
)



for k,v in output.items():

    if torch.is_tensor(v):

        print(
            k,
            v.shape
        )


print("\nCHECK COMPLETED")