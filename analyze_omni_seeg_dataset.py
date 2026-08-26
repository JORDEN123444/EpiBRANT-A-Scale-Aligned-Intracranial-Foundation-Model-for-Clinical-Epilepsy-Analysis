#!/us
#!/usr/bin/env python3

import argparse
from pathlib import Path

import mne
import pandas as pd



# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="Analyze Omni-iEEG SEEG dataset"
)


parser.add_argument(
    "--root",
    required=True
)


parser.add_argument(
    "--output",
    default="OMNI_SEEG_statistics.csv"
)


args = parser.parse_args()


ROOT = Path(args.root)

OUTPUT = args.output



# ============================================================
# Remove auxiliary channels
# ============================================================

BAD_KEYWORDS = [

    "ECG",
    "EKG",
    "EMG",
    "RESP",
    "SPO2",
    "SpO2",
    "PLETH",
    "PULSE",
    "EOG",
    "TRIG",
    "STATUS",
    "EVENT",
    "ANNOT"

]



def select_seeg_channels(ch_names):

    seeg=[]
    removed=[]


    for ch in ch_names:

        bad=False

        for key in BAD_KEYWORDS:

            if key.lower() in ch.lower():

                bad=True
                break


        if bad:
            removed.append(ch)

        else:
            seeg.append(ch)


    return seeg, removed




# ============================================================
# Extract BIDS information
# ============================================================

def extract_subject(path):

    for p in path.parts:

        if p.startswith("sub-"):

            return p


    return "unknown"



def extract_session(path):

    for p in path.parts:

        if p.startswith("ses-"):

            return p


    return "unknown"



# ============================================================
# Scan EDF
# ============================================================

edf_files=list(ROOT.rglob("*.edf"))


print("="*70)
print("Omni-iEEG SEEG analysis")
print("EDF files:",len(edf_files))
print("="*70)



records=[]



for i,f in enumerate(edf_files,1):


    print(
        f"\n[{i}/{len(edf_files)}]",
        f.name
    )


    try:


        raw=mne.io.read_raw_edf(
            f,
            preload=False,
            verbose=False
        )


        fs=float(raw.info["sfreq"])


        duration=float(raw.times[-1])


        all_channels=raw.ch_names



        seeg_channels, removed_channels = (
            select_seeg_channels(all_channels)
        )



        subject=extract_subject(f)

        session=extract_session(f)



        print(
            "Subject:",
            subject,
            "Session:",
            session
        )

        print(
            "FS:",
            fs
        )

        print(
            "Channels:",
            len(seeg_channels)
        )



        records.append({

            "dataset":
                "Omni_iEEG_SEEG",

            "subject_id":
                subject,

            "session_id":
                session,

            "file":
                str(f),

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

        })



    except Exception as e:


        print("FAILED:",f)

        print(e)




# ============================================================
# Save
# ============================================================


df=pd.DataFrame(records)


df.to_csv(
    OUTPUT,
    index=False
)



print("\n")
print("="*70)
print("SUMMARY")
print("="*70)


print(
    "Recordings:",
    len(df)
)


print(
    "Subjects:",
    df.subject_id.nunique()
)



print("\nSampling frequency")
print(
    df.sampling_rate.value_counts()
)



print("\nTotal channels")
print(
    df.total_channels.describe()
)



print("\nSEEG channels")
print(
    df.seeg_channels.describe()
)



print("\nTotal hours")
print(
    df.duration_hours.sum()
)



print("\nSaved:")
print(
    OUTPUT
)