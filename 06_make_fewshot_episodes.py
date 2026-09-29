
import argparse

import hashlib

import re

from pathlib import Path



import numpy as np

import pandas as pd





def safe_name(value):

    value = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(value))

    return value.strip("_")





def deterministic_seed(patient, shot, seed):

    text = "{}|{}|{}".format(patient, shot, seed)

    digest = hashlib.md5(text.encode("utf-8")).hexdigest()

    return int(digest[:8], 16)





def nonoverlapping_rows(df):

    """

    Keep temporally non-overlapping 60-second windows.



    The original manifest uses a 30-second stride. This function prevents

    adjacent overlapping windows from being counted as independent shots.

    """

    if df.empty:

        return df.copy()



    df = df.sort_values(

        ["segment_start_sec", "segment_end_sec", "window_id"]

    ).reset_index(drop=True)



    selected_indices = []

    previous_end = -np.inf



    for idx, row in df.iterrows():

        start = float(row["segment_start_sec"])

        end = float(row["segment_end_sec"])



        if start >= previous_end - 1e-6:

            selected_indices.append(idx)

            previous_end = end



    return df.iloc[selected_indices].reset_index(drop=True)





def choose_support_file(class_df, shot, seed):

    """

    Select one EDF as the support recording.



    Query windows must come from other EDF recordings.

    """

    eligible = []



    for raw_file, file_df in class_df.groupby("raw_file"):

        support_pool = nonoverlapping_rows(file_df)

        query_pool = class_df[class_df["raw_file"] != raw_file]



        if len(support_pool) >= shot and len(query_pool) > 0:

            eligible.append(str(raw_file))



    if not eligible:

        return None



    eligible = sorted(eligible)

    return eligible[seed % len(eligible)]





def sample_support(pool, shot, rng):

    pool = nonoverlapping_rows(pool)



    if len(pool) < shot:

        raise RuntimeError(

            "Only {} non-overlapping support windows available, but {} required."

            .format(len(pool), shot)

        )



    indices = rng.choice(len(pool), size=shot, replace=False)

    indices = np.sort(indices)



    return pool.iloc[indices].reset_index(drop=True)





def create_nonoverlapping_query(query_df):

    parts = []



    for raw_file, file_df in query_df.groupby("raw_file"):

        part = nonoverlapping_rows(file_df)

        if not part.empty:

            parts.append(part)



    if not parts:

        return pd.DataFrame(columns=query_df.columns)



    return pd.concat(parts, ignore_index=True)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)

    parser.add_argument("--lopo_runs", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--shots", nargs="+", type=int, default=[3, 5])

    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])

    args = parser.parse_args()



    manifest = pd.read_csv(args.manifest)

    lopo_runs = Path(args.lopo_runs)

    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    required = [

        "patient_id", "label", "raw_file", "cache_path",

        "segment_start_sec", "segment_end_sec", "window_id"

    ]



    missing = [column for column in required if column not in manifest.columns]

    if missing:

        raise RuntimeError("Manifest columns missing: {}".format(missing))



    patients = sorted(manifest["patient_id"].astype(str).unique())

    episode_rows = []



    print("Patients:", patients)

    print("Shots:", args.shots)

    print("Seeds:", args.seeds)



    for patient in patients:

        patient_df = manifest[

            manifest["patient_id"].astype(str) == patient

        ].copy()



        if set(patient_df["label"].astype(int).unique()) != {0, 1}:

            print("SKIP {}: patient does not contain both classes.".format(patient))

            continue



        base_run = lopo_runs / ("lopo_val_" + safe_name(patient))

        base_checkpoint = base_run / "best_model.pt"



        if not base_checkpoint.exists():

            print("SKIP {}: base checkpoint missing: {}".format(

                patient, base_checkpoint

            ))

            continue



        for shot in args.shots:

            for seed in args.seeds:

                rng = np.random.RandomState(

                    deterministic_seed(patient, shot, seed)

                )



                support_parts = []

                query_parts = []

                support_files = {}



                valid_episode = True



                for label in [0, 1]:

                    class_df = patient_df[

                        patient_df["label"].astype(int) == label

                    ].copy()



                    support_file = choose_support_file(

                        class_df=class_df,

                        shot=shot,

                        seed=seed

                    )



                    if support_file is None:

                        print(

                            "SKIP patient={} shot={} seed={}: "

                            "no suitable support EDF for label={}"

                            .format(patient, shot, seed, label)

                        )

                        valid_episode = False

                        break



                    support_pool = class_df[

                        class_df["raw_file"].astype(str) == support_file

                    ].copy()



                    support = sample_support(

                        support_pool,

                        shot=shot,

                        rng=rng

                    )



                    query_pool = class_df[

                        class_df["raw_file"].astype(str) != support_file

                    ].copy()



                    query = create_nonoverlapping_query(query_pool)



                    if query.empty:

                        print(

                            "SKIP patient={} shot={} seed={}: "

                            "query set empty for label={}"

                            .format(patient, shot, seed, label)

                        )

                        valid_episode = False

                        break



                    support_files[label] = support_file

                    support_parts.append(support)

                    query_parts.append(query)



                if not valid_episode:

                    continue



                support_df = pd.concat(

                    support_parts, ignore_index=True

                ).sample(

                    frac=1.0,

                    random_state=deterministic_seed(patient, shot, seed)

                ).reset_index(drop=True)



                query_df = pd.concat(

                    query_parts, ignore_index=True

                ).sort_values(

                    ["label", "raw_file", "segment_start_sec"]

                ).reset_index(drop=True)



                support_df["episode_split"] = "support"

                query_df["episode_split"] = "query"



                support_raw_files = set(

                    support_df["raw_file"].astype(str).unique()

                )

                query_raw_files = set(

                    query_df["raw_file"].astype(str).unique()

                )



                overlap = support_raw_files.intersection(query_raw_files)

                if overlap:

                    raise RuntimeError(

                        "Support/query EDF leakage detected: {}".format(overlap)

                    )



                episode_dir = (

                    out_dir

                    / "episodes"

                    / "k{}".format(shot)

                    / "seed{}".format(seed)

                    / safe_name(patient)

                )

                episode_dir.mkdir(parents=True, exist_ok=True)



                support_path = episode_dir / "support.csv"

                query_path = episode_dir / "query.csv"



                support_df.to_csv(support_path, index=False)

                query_df.to_csv(query_path, index=False)



                episode_rows.append({

                    "patient_id": patient,

                    "shot": int(shot),

                    "seed": int(seed),

                    "base_run": str(base_run),

                    "base_checkpoint": str(base_checkpoint),

                    "support_csv": str(support_path),

                    "query_csv": str(query_path),

                    "support_awake": int(

                        (support_df["label"].astype(int) == 0).sum()

                    ),

                    "support_sleep": int(

                        (support_df["label"].astype(int) == 1).sum()

                    ),

                    "query_awake": int(

                        (query_df["label"].astype(int) == 0).sum()

                    ),

                    "query_sleep": int(

                        (query_df["label"].astype(int) == 1).sum()

                    ),

                    "support_awake_file": support_files[0],

                    "support_sleep_file": support_files[1],

                    "result_dir": str(

                        out_dir

                        / "results"

                        / "k{}".format(shot)

                        / "seed{}".format(seed)

                        / safe_name(patient)

                    ),

                })



                print(

                    "Created patient={} shot={} seed={} | "

                    "support={} | query={}"

                    .format(

                        patient,

                        shot,

                        seed,

                        len(support_df),

                        len(query_df)

                    )

                )



    episode_table = pd.DataFrame(episode_rows)

    episode_table_path = out_dir / "fewshot_episode_table.csv"

    episode_table.to_csv(episode_table_path, index=False)



    print("\nSaved:")

    print(episode_table_path)

    print("Total episodes:", len(episode_table))



    if episode_table.empty:

        raise RuntimeError("No few-shot episodes were generated.")





if __name__ == "__main__":

    main()

