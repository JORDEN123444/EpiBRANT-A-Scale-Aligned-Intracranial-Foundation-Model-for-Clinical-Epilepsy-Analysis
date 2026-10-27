import os
import sys

import torch


# ======================================================
# Add project root
# ======================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

sys.path.append(PROJECT_ROOT)



from epibrant_pretrain_dataset import (
    EpiBRANTPretrainDataset
)

from models.scale_alignment import (
    ScaleAlignmentEmbedding
)



# ======================================================
# Configuration
# ======================================================

MANIFEST = (
    "global_pretrain_manifest.csv"
)


DEVICE = "cuda"


print("==============================")
print("DEVICE")
print("==============================")

print(DEVICE)



# ======================================================
# Load dataset
# ======================================================

dataset = EpiBRANTPretrainDataset(

    MANIFEST

)



sample = dataset[0]



signal = sample["signal"]

fs = sample["fs"]



print("\n==============================")
print("RAW SEEG INPUT")
print("==============================")


print(
    "Signal shape:",
    signal.shape
)


print(
    "Sampling rate:",
    fs
)


if "dataset" in sample:

    print(
        "Dataset:",
        sample["dataset"]
    )


print(
    "Subject:",
    sample["subject"]
)



# ======================================================
# Validate signal
# ======================================================


assert signal.ndim == 2, (
    "Expected C x T signal"
)



channels, samples = signal.shape


print("\nChannels:", channels)

print("Samples:", samples)



expected_samples = int(
    fs * 60
)


assert samples == expected_samples, (

    f"Expected 60 sec samples={expected_samples}, "
    f"got {samples}"

)



# ======================================================
# Add batch dimension
# ======================================================


# C,T
#
# ->
#
# B,C,T


signal = signal.unsqueeze(0)



signal = signal.to(
    DEVICE
)
if DEVICE == "cuda":
    torch.cuda.empty_cache()


print("\n==============================")
print("MODEL INPUT")
print("==============================")


print(
    signal.shape
)



# ======================================================
# Scale Alignment Encoder
# ======================================================


scale_encoder = ScaleAlignmentEmbedding(

    d_model=1024,

    freq_bins=128

).to(DEVICE)



scale_encoder.eval()



if DEVICE == "cuda":
    torch.cuda.empty_cache()


with torch.inference_mode():

    embeddings = scale_encoder(
        signal,
        fs
    )


print("\n==============================")
print("SCALE ALIGNMENT OUTPUT")
print("==============================")


print(
    "Embedding:",
    embeddings.shape
)



# Verify output

B,C,K,D = embeddings.shape


assert B == 1

assert C == channels

assert K == 60

assert D == 1024



# ======================================================
# Vectorized Mask Generator
# ======================================================


def generate_mask(
        x,
        mask_ratio=0.4
):

    """

    Input:

        B,C,K,D


    Output:

        B,C,K

    """

    B,C,K,D = x.shape


    random_values = torch.rand(
        B,
        C,
        K,
        device=x.device
    )


    mask = (
        random_values < mask_ratio
    )


    return mask




mask = generate_mask(
    embeddings,
    mask_ratio=0.4
)



print("\n==============================")
print("MASK")
print("==============================")


print(
    "Mask shape:",
    mask.shape
)


print(
    "Actual masking ratio:",
    mask.float().mean().item()
)



# ======================================================
# GPU Memory
# ======================================================

if DEVICE=="cuda":

    print("\n==============================")
    print("GPU MEMORY")
    print("==============================")


    print(
        "Allocated GB:",
        torch.cuda.memory_allocated()/1024**3
    )


    print(
        "Reserved GB:",
        torch.cuda.memory_reserved()/1024**3
    )



print("\n==============================")
print("SUCCESS")
print("==============================")