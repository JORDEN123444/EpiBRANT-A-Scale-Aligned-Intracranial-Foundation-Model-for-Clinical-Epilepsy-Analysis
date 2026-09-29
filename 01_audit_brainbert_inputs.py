
import argparse

from pathlib import Path



import numpy as np

import pandas as pd





SIGNAL_SUFFIXES = (

    ".edf", ".bdf", ".fif", ".vhdr", ".set",

    ".npy", ".npz", ".h5", ".hdf5"

)





def audit(name, path):

    path = Path(path)



    print("\n" + "=" * 110)

    print(name)

    print(path)

    print("=" * 110)



    if not path.exists():

        print("MISSING FILE")

        return



    df = pd.read_csv(path)



    print("Rows:", len(df))

    print("Columns:")

    print(df.columns.tolist())



    for column in [

        "state",

        "label",

        "patient_id",

        "segment_id",

        "source_dataset",

    ]:

        if column in df.columns:

            print(f"\n{column}:")

            print(

                df[column]

                .value_counts(dropna=False)

                .head(40)

                .to_string()

            )



    print("\nPossible signal and sidecar paths:")



    for column in df.columns:

        if df[column].dtype != object:

            continue



        values = df[column].dropna().astype(str)



        matching = values[

            values.str.lower().str.endswith(SIGNAL_SUFFIXES)

        ]



        if not matching.empty:

            existing = sum(

                Path(value).exists()

                for value in matching

            )



            print(

                f"{column}: matching={len(matching)}, "

                f"existing={existing}, example={matching.iloc[0]}"

            )



    for token in [

        "channels",

        "electrodes",

        "coordinate",

        "start",

        "end",

        "offset",

        "sampling",

        "sfreq",

        "fs",

    ]:

        matched_columns = [

            column

            for column in df.columns

            if token in column.lower()

        ]



        if matched_columns:

            print(f"\nColumns containing '{token}':")

            for column in matched_columns:

                print(column, df[column].head(3).tolist())





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seeg", required=True)

    parser.add_argument("--omni", required=True)

    args = parser.parse_args()



    audit("SEEG dataset", args.seeg)

    audit("Omni-iEEG dataset", args.omni)





if __name__ == "__main__":

    main()

