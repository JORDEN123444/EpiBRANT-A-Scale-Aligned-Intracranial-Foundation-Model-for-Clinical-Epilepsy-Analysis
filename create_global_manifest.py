import os
from pathlib import Path

import mne
import pandas as pd


# ==========================================================
# PATHS
# ==========================================================

INSTITUTIONAL_SEEG = (
    "/home/ubuntu/Ijaz/SEEG datast/"
    "pretrainingdataset/newSEEG data"
)


OMNI_SEEG = (
    "/home/ubuntu/Ijaz/SEEG datast/"
    "pretrainingdataset/"
    "seeg-OMNI-seeg FINE TUNE 70 PATEINST"
)


OUTPUT = "global_pretrain_manifest.csv"



# ==========================================================
# Remove auxiliary channels
# ==========================================================

BAD_KEYWORDS = [

    "ECG",
    "EKG",
    "EKG1",
    "EMG",
    "EOG",
    "RESP",
    "SPO2",
    "PLETH",
    "PULSE",
    "STATUS",
    "TRIG",
    "EVENT",
    "ANNOT"

]



def select_seeg_channels(channels):

    keep=[]
    removed=[]


    for ch in channels:

        flag=False


        for key in BAD_KEYWORDS:

            if key.lower() in ch.lower():

                flag=True
                break


        if flag:

            removed.append(ch)

        else:

            keep.append(ch)



    return keep, removed




# ==========================================================
# EDF analysis
# ==========================================================


def analyze_edf(edf_file):


    raw = mne.io.read_raw_edf(

        edf_file,

        preload=False,

        verbose=False,

        encoding="latin1",

        infer_types=True

    )


    fs=float(
        raw.info["sfreq"]
    )


    duration=float(
        raw.n_times/fs
    )


    channels=raw.ch_names


    seeg_channels, removed_channels = (
        select_seeg_channels(channels)
    )


    return {


        "sampling_rate":

            fs,


        "duration_seconds":

            duration,


        "duration_hours":

            duration/3600,


        "total_channels":

            len(channels),


        "seeg_channels":

            len(seeg_channels),


        "removed_channels":

            len(removed_channels),


        "seeg_channel_names":

            ";".join(seeg_channels),


        "removed_channel_names":

            ";".join(removed_channels)

    }




# ==========================================================
# Institutional SEEG
#
# /newSEEG data/
#
#       63/
#          1.edf
#
# ==========================================================


def collect_institutional():


    records=[]


    files=list(
        Path(INSTITUTIONAL_SEEG)
        .rglob("*.edf")
    )


    print(
        "Institutional EDF:",
        len(files)
    )


    for edf in files:


        print(
            "Processing:",
            edf
        )


        info=analyze_edf(
            str(edf)
        )


        subject_id = edf.parent.name


        session_id = edf.stem



        records.append({


            "dataset":
                "Institutional_SEEG",


            "subject_id":
                subject_id,


            "session_id":
                session_id,


            "recording_id":
                edf.stem,


            "file":
                str(edf),


            **info


        })



    return records




# ==========================================================
# Omni-iEEG
#
# BIDS:
#
# sub-XXX/
#       ses-XXX/
#              ieeg/
#                 *.edf
#
# ==========================================================


def extract_bids_information(edf):


    subject_id=None

    session_id=None



    for parent in edf.parents:


        name=parent.name



        if name.startswith("sub-"):


            subject_id=name.replace(
                "sub-",
                ""
            )


        if name.startswith("ses-"):


            session_id=name



    if subject_id is None:


        raise RuntimeError(
            f"Cannot extract subject from {edf}"
        )


    if session_id is None:

        session_id="unknown"



    return subject_id, session_id





def collect_omni():


    records=[]


    files=list(
        Path(OMNI_SEEG)
        .rglob("*.edf")
    )


    print(
        "Omni EDF:",
        len(files)
    )



    for edf in files:


        print(
            "Processing:",
            edf
        )



        info=analyze_edf(
            str(edf)
        )



        subject_id, session_id = (
            extract_bids_information(edf)
        )



        records.append({


            "dataset":

                "Omni_iEEG_SEEG",



            "subject_id":

                subject_id,



            "session_id":

                session_id,



            "recording_id":

                edf.stem,



            "file":

                str(edf),



            **info


        })



    return records





# ==========================================================
# MAIN
# ==========================================================


if __name__=="__main__":



    records=[]



    records.extend(
        collect_institutional()
    )


    records.extend(
        collect_omni()
    )



    df=pd.DataFrame(records)



    # Remove very short recordings

    before=len(df)


    df=df[
        df.duration_seconds > 10
    ]


    print(
        "Removed recordings:",
        before-len(df)
    )



    # Save

    df.to_csv(

        OUTPUT,

        index=False

    )



    print("\n========================")

    print("GLOBAL MANIFEST")

    print("========================")


    print(df.head())



    print("\nDataset distribution")

    print(
        df.dataset.value_counts()
    )



    print("\nSubjects")

    print(
        df.groupby(
            "dataset"
        )
        .subject_id
        .nunique()
    )



    print("\nSessions")

    print(
        df.groupby(
            "dataset"
        )
        .session_id
        .nunique()
    )



    print("\nTotal hours")

    print(
        df.duration_hours.sum()
    )


    print("\nSaved:")

    print(
        OUTPUT
    )