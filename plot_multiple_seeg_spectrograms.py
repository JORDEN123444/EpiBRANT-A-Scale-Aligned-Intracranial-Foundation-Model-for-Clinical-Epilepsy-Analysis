import os
from pathlib import Path

import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import signal


# ======================================================
# DATASET PATH
# ======================================================

ROOT = Path(
"/home/ubuntu/Ijaz/SEEG datast/pretrainingdataset/newSEEG data"
)


OUTPUT = "SEEG_multiple_spectrograms.png"



# ======================================================
# Select EDF files
# ======================================================

edf_files = sorted(
    ROOT.rglob("*.edf")
)


# Select different patients

selected_files = [
    edf_files[0],
    edf_files[20],
    edf_files[50],
    edf_files[100],
    edf_files[150],
    edf_files[200]
]



print("Selected files:")

for f in selected_files:
    print(f)



# ======================================================
# Plot
# ======================================================


fig, axes = plt.subplots(
    2,
    3,
    figsize=(7,4),
    dpi=600
)



axes = axes.flatten()



for ax, file in zip(
    axes,
    selected_files
):


    raw = mne.io.read_raw_edf(
        file,
        preload=True,
        verbose=False
    )


    fs = raw.info["sfreq"]


    # first SEEG contact

    x = raw.get_data()[0]


    length=int(
        min(
            60*fs,
            len(x)
        )
    )


    x=x[:length]


    # filtering

    x=mne.filter.filter_data(
        x,
        sfreq=fs,
        l_freq=0.5,
        h_freq=min(120,fs/2-1),
        verbose=False
    )



    # spectrogram

    f,t,S = signal.spectrogram(
        x,
        fs=fs,
        nperseg=int(fs*2),
        noverlap=int(fs)
    )


    mask=f<=120


    S=10*np.log10(
        S[mask]+1e-12
    )


    f=f[mask]



    im=ax.pcolormesh(
        t,
        f,
        S,
        shading="auto",
        cmap="magma"
    )


    ax.set_ylim(
        0,
        120
    )


    ax.set_xlim(
        0,
        60
    )


    ax.tick_params(
        labelsize=6
    )


    ax.set_xlabel(
        "Time (s)",
        fontsize=7
    )


    ax.set_ylabel(
        "Frequency (Hz)",
        fontsize=7
    )



plt.tight_layout()



plt.savefig(
    OUTPUT,
    dpi=600,
    bbox_inches="tight"
)



print("Saved:",OUTPUT)