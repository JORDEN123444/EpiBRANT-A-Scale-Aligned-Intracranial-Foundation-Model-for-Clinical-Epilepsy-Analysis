from pathlib import Path
import pandas as pd
import mne


# ==========================================================
# PATHS
# ==========================================================

DATASET_ROOT = Path(
    "/home/ubuntu/Ijaz/SEEG datast/pretrainingdataset/SEEG-omnidtaset"
)


OUTPUT = Path(
    "downstream/soz_localization/manifests/soz_manifest.csv"
)


# ==========================================================
# SETTINGS
# ==========================================================

WINDOW_SECONDS = 60


MIN_SEEG_CHANNELS = 5



# ==========================================================
# CHECK SEEG CHANNELS
# ==========================================================

def count_seeg_channels(
    channels_file
):

    df = pd.read_csv(
        channels_file,
        sep="\t"
    )


    if "type" not in df.columns:

        return 0


    seeg = df[
        df["type"]
        .astype(str)
        .str.upper()
        ==
        "SEEG"
    ]


    return len(seeg)



# ==========================================================
# FIND CHANNEL TSV
# ==========================================================

def find_channels_file(
    edf_file
):

    name = edf_file.name.replace(
        "_ieeg.edf",
        "_channels.tsv"
    )


    candidate = edf_file.parent / name


    if candidate.exists():

        return candidate


    return None



# ==========================================================
# BUILD MANIFEST
# ==========================================================

def build_manifest():


    rows = []


    subjects = sorted(
        DATASET_ROOT.glob(
            "sub-*"
        )
    )


    print(
        "Subjects found:",
        len(subjects)
    )



    for sub in subjects:


        ieeg_folder = (
            sub /
            "ses-01" /
            "ieeg"
        )


        if not ieeg_folder.exists():

            continue



        edf_files = sorted(
            ieeg_folder.glob(
                "*_ieeg.edf"
            )
        )


        for edf in edf_files:


            channels_file = (
                find_channels_file(
                    edf
                )
            )


            if channels_file is None:

                print(
                    "Missing channels:",
                    edf
                )

                continue



            n_seeg = count_seeg_channels(
                channels_file
            )


            if n_seeg < MIN_SEEG_CHANNELS:

                continue



            # --------------------------------
            # Read EDF information
            # --------------------------------

            try:

                raw = mne.io.read_raw_edf(

                    edf,

                    preload=False,

                    verbose=False

                )


                fs = float(
                    raw.info["sfreq"]
                )


                n_samples = (
                    raw.n_times
                )


            except Exception as e:

                print(
                    "EDF error:",
                    edf,
                    e
                )

                continue



            window_size = int(
                WINDOW_SECONDS * fs
            )



            # --------------------------------
            # Create windows
            # --------------------------------

            start = 0


            while (
                start + window_size
                <= n_samples
            ):


                rows.append(

                    {

                    "patient_id":
                        sub.name,


                    "file":
                        str(edf),


                    "channels_file":
                        str(channels_file),


                    "sampling_rate":
                        fs,


                    "start_sample":
                        start,


                    "stop_sample":
                        start + window_size,


                    "duration_seconds":
                        WINDOW_SECONDS,


                    "num_seeg_channels":
                        n_seeg

                    }

                )


                start += window_size




    df = pd.DataFrame(
        rows
    )


    if len(df)==0:

        raise RuntimeError(
            "No SOZ samples generated"
        )


    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    df.to_csv(
        OUTPUT,
        index=False
    )


    print("="*80)

    print(
        "SOZ MANIFEST CREATED"
    )

    print(
        "Samples:",
        len(df)
    )

    print(
        "Patients:",
        df.patient_id.nunique()
    )

    print(
        "Output:",
        OUTPUT
    )

    print("="*80)



if __name__ == "__main__":

    build_manifest()