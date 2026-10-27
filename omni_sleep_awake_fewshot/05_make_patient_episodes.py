
import argparse

from pathlib import Path



import numpy as np

import pandas as pd





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



    parser.add_argument(

        "--seeds",

        nargs="+",

        type=int,

        default=[11, 23, 47, 71, 101],

    )



    parser.add_argument(

        "--query_per_class",

        type=int,

        default=3,

    )



    args = parser.parse_args()



    manifest = pd.read_csv(

        args.manifest

    )



    required = [

        "patient_id",

        "state",

        "label",

        "embedding_path",

    ]



    missing = [

        column

        for column in required

        if column not in manifest.columns

    ]



    if missing:

        raise RuntimeError(

            "Missing columns: {}".format(

                missing

            )

        )



    awake = (

        manifest[

            manifest["label"].astype(int) == 0

        ]

        .sort_values("patient_id")

        .reset_index(drop=True)

    )



    sleep = (

        manifest[

            manifest["label"].astype(int) == 1

        ]

        .sort_values("patient_id")

        .reset_index(drop=True)

    )



    if len(awake) != 8 or len(sleep) != 8:

        raise RuntimeError(

            "Expected eight patients per class."

        )



    output_dir = Path(

        args.out_dir

    )



    episode_root = (

        output_dir / "patient_episodes"

    )



    episode_root.mkdir(

        parents=True,

        exist_ok=True,

    )



    episode_rows = []



    for seed in args.seeds:

        # Separate generators prevent one class from

        # affecting the permutation of the other class.

        awake_rng = np.random.RandomState(

            seed

        )



        sleep_rng = np.random.RandomState(

            seed + 10000

        )



        awake_order = awake_rng.permutation(

            len(awake)

        )



        sleep_order = sleep_rng.permutation(

            len(sleep)

        )



        query_count = args.query_per_class



        awake_query = awake.iloc[

            awake_order[:query_count]

        ].copy()



        sleep_query = sleep.iloc[

            sleep_order[:query_count]

        ].copy()



        awake_support_pool = awake.iloc[

            awake_order[query_count:]

        ].copy()



        sleep_support_pool = sleep.iloc[

            sleep_order[query_count:]

        ].copy()



        if (

            len(awake_support_pool) != 5

            or len(sleep_support_pool) != 5

        ):

            raise RuntimeError(

                "Expected five support-pool "

                "patients per class."

            )



        query_df = pd.concat(

            [

                awake_query,

                sleep_query,

            ],

            ignore_index=True,

        )



        query_df = query_df.sample(

            frac=1.0,

            random_state=seed,

        ).reset_index(drop=True)



        for shot in [3, 5]:

            support_df = pd.concat(

                [

                    awake_support_pool.iloc[

                        :shot

                    ],

                    sleep_support_pool.iloc[

                        :shot

                    ],

                ],

                ignore_index=True,

            )



            support_df = support_df.sample(

                frac=1.0,

                random_state=seed,

            ).reset_index(drop=True)



            support_patients = set(

                support_df["patient_id"]

                .astype(str)

            )



            query_patients = set(

                query_df["patient_id"]

                .astype(str)

            )



            overlap = (

                support_patients

                .intersection(query_patients)

            )



            if overlap:

                raise RuntimeError(

                    "Patient leakage: {}".format(

                        sorted(overlap)

                    )

                )



            support_awake = int(

                (

                    support_df["label"]

                    .astype(int) == 0

                ).sum()

            )



            support_sleep = int(

                (

                    support_df["label"]

                    .astype(int) == 1

                ).sum()

            )



            query_awake_count = int(

                (

                    query_df["label"]

                    .astype(int) == 0

                ).sum()

            )



            query_sleep_count = int(

                (

                    query_df["label"]

                    .astype(int) == 1

                ).sum()

            )



            if support_awake != shot:

                raise RuntimeError(

                    "Invalid awake support count."

                )



            if support_sleep != shot:

                raise RuntimeError(

                    "Invalid sleep support count."

                )



            if query_awake_count != query_count:

                raise RuntimeError(

                    "Invalid awake query count."

                )



            if query_sleep_count != query_count:

                raise RuntimeError(

                    "Invalid sleep query count."

                )



            episode_dir = (

                episode_root

                / "seed{}".format(seed)

                / "k{}".format(shot)

            )



            episode_dir.mkdir(

                parents=True,

                exist_ok=True,

            )



            support_path = (

                episode_dir / "support.csv"

            )



            query_path = (

                episode_dir / "query.csv"

            )



            support_df.to_csv(

                support_path,

                index=False,

            )



            query_df.to_csv(

                query_path,

                index=False,

            )



            result_dir = (

                output_dir

                / "prototype_results"

                / "seed{}".format(seed)

                / "k{}".format(shot)

            )



            episode_rows.append({

                "seed": int(seed),

                "shot": int(shot),

                "support_csv": str(

                    support_path

                ),

                "query_csv": str(

                    query_path

                ),

                "result_dir": str(

                    result_dir

                ),

                "support_awake": support_awake,

                "support_sleep": support_sleep,

                "query_awake": query_awake_count,

                "query_sleep": query_sleep_count,

                "support_patient_ids": "|".join(

                    sorted(support_patients)

                ),

                "query_patient_ids": "|".join(

                    sorted(query_patients)

                ),

            })



            print(

                "Created seed={} | {}-shot | "

                "support={} | query={}"

                .format(

                    seed,

                    shot,

                    len(support_df),

                    len(query_df),

                )

            )



    episode_table = pd.DataFrame(

        episode_rows

    )



    output_path = (

        output_dir

        / "patient_episode_table.csv"

    )



    episode_table.to_csv(

        output_path,

        index=False,

    )



    print("\nTotal episodes:", len(episode_table))

    print("Saved:", output_path)



    if len(episode_table) != 10:

        raise RuntimeError(

            "Expected 10 total episodes."

        )





if __name__ == "__main__":

    main()

