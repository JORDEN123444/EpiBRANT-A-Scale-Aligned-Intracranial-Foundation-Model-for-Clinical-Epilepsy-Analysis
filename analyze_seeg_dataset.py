#!/usr/bin/env python3

import argparse
from pathlib import Path
import os

import mne
import pandas as pd


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Analyze SEEG/iEEG dataset statistics for EpiBRANT pretraining"
)


parser.add_argument(
    "--root",
    required=True,
    help="Dataset root directory"
)


parser.add_argument(
    "--output",
    default="SEEG_dataset_statistics.csv",
    help="Output CSV file"
)


parser.add_argument(
    "--dataset-name",
    default="SEEG",
    help="Dataset identifier"
)


args = parser.parse_args()


ROOT = Path(args.root)
OUTPUT = args.output
DATASET_NAME = args.dataset_name



# ============================================================
# CHANNEL FILTERING
# ============================================================

# IMPORTANT:
# Do NOT include POL.
# Many SEEG systems use POL in contact names.
# Example:
# POL A1, POL B2 are real SEEG contacts.

BAD_KEYWORDS = [

    # physiological auxiliary
    "ECG",
    "EKG",
    "EMG",
    "RESP",
    "SpO2",
    "SPO2",
    "PLETH",
    "PULSE",

    # eye
    "EOG",

    # triggers/events
    "TRIG",
    "STATUS",
    "EVENT",
    "ANNOT"

]



def select_seeg_channels(channel_names):

    """
    Select intracranial SEEG contacts
    and remove auxiliary channels.
    """

    seeg = []
    removed = []


    for ch in channel_names:

        name = ch.upper()


        is_bad = False


        for key in BAD_KEYWORDS:

            if key.upper() in name:
                is_bad = True
                break



        if is_bad:

            removed.append(ch)

        else:

            seeg.append(ch)



    return seeg, removed



# ============================================================
# SUBJECT EXTRACTION
# ============================================================

def extract_subject_id(path):

    """
    Extract subject ID from common structures:

    sub-XXXXX/session/file.edf

    or

    patient001/file.edf
    """

    parts = list(path.parts)


    for p in parts:

        if p.startswith("sub-"):

            return p.replace("sub-", "")


    for p in parts:

        if (
            "patient" in p.lower()
            or "subject" in p.lower()
        ):

            return p



    return "unknown"



# ============================================================
# FIND EDF FILES
# ============================================================


edf_files = list(
    ROOT.rglob("*.edf")
)


print("="*70)
print("Dataset:", DATASET_NAME)
print("Root:", ROOT)
print("EDF files:", len(edf_files))
print("="*70)



records = []



# ============================================================
# PROCESS FILES
# ============================================================


for idx, file in enumerate(edf_files,1):


    print(
        f"\n[{idx}/{len(edf_files)}] Processing:"
        ,
        file.name
    )


    try:


        raw = mne.io.read_raw_edf(
            file,
            preload=False,
            verbose=False
        )


        fs = float(
            raw.info["sfreq"]
        )


        duration = float(
            raw.times[-1]
        )


        all_channels = raw.ch_names


        seeg_channels, removed_channels = (
            select_seeg_channels(all_channels)
        )


        subject_id = extract_subject_id(file)



        print(
            "Subject:",
            subject_id
        )

        print(
            "Sampling:",
            fs,
            "Hz"
        )

        print(
            "Total channels:",
            len(all_channels)
        )


        print(
            "SEEG channels:",
            len(seeg_channels)
        )


        print(
            "Removed:",
            len(removed_channels)
        )



        records.append(

            {

            "dataset":
                DATASET_NAME,


            "subject_id":
                subject_id,


            "file":
                str(file),


            "sampling_rate":
                fs,


            "duration_seconds":
                duration,


            "duration_hours":
                duration/3600,


            "total_channels":
                len(all_channels),


            "seeg_channels":
                len(seeg_channels),


            "removed_channels":
                len(removed_channels),


            "seeg_channel_names":
                ";".join(seeg_channels),


            "removed_channel_names":
                ";".join(removed_channels)

            }

        )



    except Exception as e:


        print("\nFAILED:")
        print(file)

        print(e)




# ============================================================
# SAVE CSV
# ============================================================


df = pd.DataFrame(records)


df.to_csv(
    OUTPUT,
    index=False
)



# ============================================================
# SUMMARY
# ============================================================


print("\n")
print("="*70)
print("FINAL SUMMARY")
print("="*70)


print(
    "Dataset:",
    DATASET_NAME
)


print(
    "Recordings:",
    len(df)
)



print("\nSubjects:")
print(
    df.subject_id.nunique()
)



print("\nSampling frequency:")
print(
    df.sampling_rate.value_counts()
)



print("\nTotal channels:")
print(
    df.total_channels.describe()
)



print("\nSEEG channels:")
print(
    df.seeg_channels.describe()
)



print("\nTotal hours:")
print(
    df.duration_hours.sum()
)



print("\nSaved:")
print(
    OUTPUT
)

print("="*70)