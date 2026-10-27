
import argparse

from pathlib import Path



import numpy as np

import pandas as pd





def l2_normalize(

    array,

    axis=-1,

    epsilon=1e-12,

):

    norms = np.linalg.norm(

        array,

        axis=axis,

        keepdims=True,

    )



    norms = np.maximum(

        norms,

        epsilon,

    )



    return array / norms





def add_missing_metadata(

    embedding_manifest,

    out_dir,

):

    required_metadata = [

        "patient_id",

        "state",

        "label",

        "segment_id",

    ]



    missing = [

        column

        for column in required_metadata

        if column not in embedding_manifest.columns

    ]



    if not missing:

        return embedding_manifest



    segment_manifest_path = (

        out_dir / "segment_manifest.csv"

    )



    if not segment_manifest_path.exists():

        raise FileNotFoundError(

            segment_manifest_path

        )



    segment_manifest = pd.read_csv(

        segment_manifest_path

    )



    merge_columns = [

        "segment_id",

        "patient_id",

        "state",

        "label",

    ]



    optional_columns = [

        "original_split",

        "split",

        "source_dataset",

        "pair_id",

        "edf_path",

        "raw_file",

    ]



    merge_columns.extend(

        [

            column

            for column in optional_columns

            if column in segment_manifest.columns

        ]

    )



    metadata = (

        segment_manifest[

            merge_columns

        ]

        .drop_duplicates(

            subset=["segment_id"]

        )

    )



    columns_to_remove = [

        column

        for column in metadata.columns

        if (

            column != "segment_id"

            and column in embedding_manifest.columns

        )

    ]



    metadata = metadata.drop(

        columns=columns_to_remove

    )



    result = embedding_manifest.merge(

        metadata,

        on="segment_id",

        how="left",

        validate="one_to_one",

    )



    return result





def unique_value(

    dataframe,

    column,

):

    if column not in dataframe.columns:

        return None



    values = (

        dataframe[column]

        .dropna()

        .astype(str)

        .unique()

    )



    if len(values) == 0:

        return None



    if len(values) > 1:

        raise RuntimeError(

            "Multiple {} values were found: {}"

            .format(

                column,

                values,

            )

        )



    return values[0]





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--manifest",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    args = parser.parse_args()



    manifest_path = Path(

        args.manifest

    )



    out_dir = Path(

        args.out_dir

    )



    if not manifest_path.exists():

        raise FileNotFoundError(

            manifest_path

        )



    manifest = pd.read_csv(

        manifest_path

    )



    manifest = add_missing_metadata(

        manifest,

        out_dir,

    )



    required_columns = [

        "patient_id",

        "state",

        "label",

        "embedding_path",

    ]



    missing_columns = [

        column

        for column in required_columns

        if column not in manifest.columns

    ]



    if missing_columns:

        raise RuntimeError(

            "Missing required columns: {}"

            .format(missing_columns)

        )



    patient_embedding_dir = (

        out_dir / "patient_embeddings"

    )



    patient_embedding_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    output_rows = []



    for patient_id, patient_df in manifest.groupby(

        "patient_id"

    ):

        labels = (

            patient_df["label"]

            .astype(int)

            .unique()

        )



        states = (

            patient_df["state"]

            .astype(str)

            .unique()

        )



        if len(labels) != 1:

            raise RuntimeError(

                "Patient {} has multiple labels: {}"

                .format(

                    patient_id,

                    labels,

                )

            )



        if len(states) != 1:

            raise RuntimeError(

                "Patient {} has multiple states: {}"

                .format(

                    patient_id,

                    states,

                )

            )



        embeddings = []



        for path_value in patient_df[

            "embedding_path"

        ].astype(str):

            path = Path(path_value)



            if not path.exists():

                raise FileNotFoundError(path)



            embedding = np.load(

                path

            ).astype(np.float32)



            if embedding.shape != (192,):

                raise RuntimeError(

                    "Expected (192,), found {} in {}."

                    .format(

                        embedding.shape,

                        path,

                    )

                )



            if not np.isfinite(

                embedding

            ).all():

                raise RuntimeError(

                    "Invalid embedding values in {}."

                    .format(path)

                )



            embeddings.append(

                embedding

            )



        window_embeddings = np.stack(

            embeddings,

            axis=0,

        )



        normalized_windows = l2_normalize(

            window_embeddings,

            axis=1,

        )



        patient_embedding = np.mean(

            normalized_windows,

            axis=0,

        )



        patient_embedding = l2_normalize(

            patient_embedding[

                np.newaxis,

                :

            ],

            axis=1,

        )[0].astype(np.float32)



        patient_path = (

            patient_embedding_dir

            / "{}.npy".format(

                patient_id

            )

        )



        np.save(

            patient_path,

            patient_embedding,

        )



        output_row = {

            "patient_id": str(patient_id),

            "state": str(states[0]),

            "label": int(labels[0]),

            "number_of_windows": int(

                len(patient_df)

            ),

            "embedding_dimension": int(

                patient_embedding.shape[0]

            ),

            "embedding_path": str(

                patient_path

            ),

            "patient_embedding_path": str(

                patient_path

            ),

        }



        for column in [

            "original_split",

            "split",

            "source_dataset",

            "pair_id",

            "edf_path",

            "raw_file",

        ]:

            value = unique_value(

                patient_df,

                column,

            )



            if value is not None:

                output_row[column] = value



        output_rows.append(

            output_row

        )



    patient_manifest = pd.DataFrame(

        output_rows

    ).sort_values(

        [

            "label",

            "patient_id",

        ]

    )



    output_path = (

        out_dir

        / "patient_embedding_manifest.csv"

    )



    patient_manifest.to_csv(

        output_path,

        index=False,

    )



    print("\nPATIENT EMBEDDING SUMMARY")

    print("=" * 100)



    print(

        patient_manifest[

            [

                "patient_id",

                "state",

                "label",

                "number_of_windows",

                "embedding_dimension",

            ]

        ].to_string(index=False)

    )



    print("\nClass counts:")

    print(

        patient_manifest.groupby(

            "state"

        )["patient_id"].nunique()

    )



    print("\nTotal patients:", len(patient_manifest))

    print("Saved:", output_path)



    if len(patient_manifest) != 16:

        raise RuntimeError(

            "Expected 16 patients, found {}."

            .format(len(patient_manifest))

        )



    counts = (

        patient_manifest["label"]

        .value_counts()

        .to_dict()

    )



    if counts.get(0, 0) != 8:

        raise RuntimeError(

            "Expected eight awake patients."

        )



    if counts.get(1, 0) != 8:

        raise RuntimeError(

            "Expected eight sleep patients."

        )





if __name__ == "__main__":

    main()

