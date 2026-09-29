
import argparse

from pathlib import Path



import numpy as np

import pandas as pd

from sklearn.metrics import (

    accuracy_score,

    average_precision_score,

    balanced_accuracy_score,

    cohen_kappa_score,

    confusion_matrix,

    f1_score,

    matthews_corrcoef,

    precision_score,

    roc_auc_score,

)





SEEDS = [11, 23, 47, 71, 101]





def load_embedding(path):

    embedding = np.load(

        path,

        allow_pickle=False,

    ).astype(np.float64)



    norm = np.linalg.norm(embedding)



    return embedding / norm if norm > 0 else embedding





def classify(support_x, support_y, query_x):

    prototypes = []



    for label in [0, 1]:

        prototype = support_x[

            support_y == label

        ].mean(axis=0)



        norm = np.linalg.norm(prototype)



        if norm > 0:

            prototype = prototype / norm



        prototypes.append(prototype)



    prototypes = np.stack(prototypes)



    similarities = query_x @ prototypes.T

    predictions = similarities.argmax(axis=1)



    score = similarities[:, 1] - similarities[:, 0]



    return predictions, score





def safe_divide(numerator, denominator):
    if denominator == 0:
        return np.nan

    return float(numerator / denominator)


def calculate_metrics(y_true, y_pred, score):
    y_true = np.asarray(
        y_true,
        dtype=int,
    )

    y_pred = np.asarray(
        y_pred,
        dtype=int,
    )

    score = np.asarray(
        score,
        dtype=float,
    )

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    ).ravel()

    # Sleep is the positive class.
    sensitivity_sleep = safe_divide(
        tp,
        tp + fn,
    )

    specificity_awake = safe_divide(
        tn,
        tn + fp,
    )

    precision_sleep = safe_divide(
        tp,
        tp + fp,
    )

    # Precision for class 0. This is also the negative predictive value.
    precision_awake = safe_divide(
        tn,
        tn + fn,
    )

    npv = precision_awake

    false_positive_rate = safe_divide(
        fp,
        fp + tn,
    )

    false_negative_rate = safe_divide(
        fn,
        fn + tp,
    )

    geometric_mean = (
        np.sqrt(
            sensitivity_sleep
            * specificity_awake
        )
        if (
            np.isfinite(sensitivity_sleep)
            and np.isfinite(specificity_awake)
        )
        else np.nan
    )

    youden_j = (
        sensitivity_sleep
        + specificity_awake
        - 1.0
        if (
            np.isfinite(sensitivity_sleep)
            and np.isfinite(specificity_awake)
        )
        else np.nan
    )

    unique_classes = np.unique(
        y_true
    )

    auroc = (
        roc_auc_score(
            y_true,
            score,
        )
        if len(unique_classes) == 2
        else np.nan
    )

    auprc = (
        average_precision_score(
            y_true,
            score,
        )
        if len(unique_classes) == 2
        else np.nan
    )

    return {
        # Overall performance
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred,
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),

        # Sleep-class performance
        "precision_sleep": precision_sleep,
        "sensitivity_sleep": sensitivity_sleep,
        "recall_sleep": sensitivity_sleep,

        "f1_sleep": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),

        # Awake-class performance
        "precision_awake": precision_awake,
        "specificity_awake": specificity_awake,
        "recall_awake": specificity_awake,

        "f1_awake": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=0,
                zero_division=0,
            )
        ),

        # Averaged F1 scores
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            )
        ),

        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0,
            )
        ),

        # Correlation and agreement
        "mcc": float(
            matthews_corrcoef(
                y_true,
                y_pred,
            )
        ),

        "kappa": float(
            cohen_kappa_score(
                y_true,
                y_pred,
            )
        ),

        # Ranking metrics
        "auroc": float(auroc),
        "auprc": float(auprc),

        # Additional diagnostic metrics
        "negative_predictive_value": npv,
        "geometric_mean": geometric_mean,
        "youden_j": youden_j,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,

        # Confusion matrix counts
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),

        # Class sample counts
        "support_awake": int(
            np.sum(y_true == 0)
        ),

        "support_sleep": int(
            np.sum(y_true == 1)
        ),
    }





def within_patient(df):

    rows = []

    assignments = []



    for patient_id, patient_df in df.groupby(

        "patient_id"

    ):

        class_indices = {

            label: patient_df.index[

                patient_df["label"] == label

            ].to_numpy()

            for label in [0, 1]

        }



        if any(len(class_indices[label]) < 7 for label in [0, 1]):

            print(

                "Skipping patient with insufficient samples:",

                patient_id,

            )

            continue



        for seed in SEEDS:

            rng = np.random.default_rng(seed)



            permutations = {

                label: rng.permutation(

                    class_indices[label]

                )

                for label in [0, 1]

            }



            # Fixed query set for both shot settings.

            query_indices = np.concatenate([

                permutations[0][-2:],

                permutations[1][-2:],

            ])



            for shot in [3, 5]:

                support_indices = np.concatenate([

                    permutations[0][:shot],

                    permutations[1][:shot],

                ])



                support = df.loc[support_indices]

                query = df.loc[query_indices]



                support_x = np.stack([

                    load_embedding(path)

                    for path in support["embedding_path"]

                ])



                support_y = support["label"].to_numpy(

                    dtype=int

                )



                query_x = np.stack([

                    load_embedding(path)

                    for path in query["embedding_path"]

                ])



                query_y = query["label"].to_numpy(

                    dtype=int

                )



                predictions, score = classify(

                    support_x,

                    support_y,

                    query_x,

                )



                metrics = calculate_metrics(

                    query_y,

                    predictions,

                    score,

                )



                rows.append({

                    "dataset": "SEEG",

                    "model": "BrainBERT",

                    "patient_id": patient_id,

                    "seed": seed,

                    "shot": shot,

                    **metrics,

                })



                assignments.append({

                    "dataset": "SEEG",

                    "patient_id": patient_id,

                    "seed": seed,

                    "shot": shot,

                    "support_segment_ids": "|".join(

                        support["segment_id"].astype(str)

                    ),

                    "query_segment_ids": "|".join(

                        query["segment_id"].astype(str)

                    ),

                })



    results = pd.DataFrame(rows)

    assignments = pd.DataFrame(assignments)



    macro = (

        results

        .groupby(

            ["dataset", "model", "seed", "shot"],

            as_index=False,

        )

        .mean(numeric_only=True)

    )



    return results, macro, assignments





def cross_patient(df):

    patient_rows = []



    for patient_id, patient_df in df.groupby(

        "patient_id"

    ):

        labels = patient_df["label"].unique()



        if len(labels) != 1:

            raise RuntimeError(

                f"Omni patient {patient_id} has multiple labels."

            )



        embeddings = np.stack([

            load_embedding(path)

            for path in patient_df["embedding_path"]

        ])



        embedding = embeddings.mean(axis=0)

        norm = np.linalg.norm(embedding)



        if norm > 0:

            embedding = embedding / norm



        patient_rows.append({

            "patient_id": patient_id,

            "label": int(labels[0]),

            "embedding": embedding,

        })



    patient_df = pd.DataFrame(patient_rows)



    class_patients = {

        label: patient_df.index[

            patient_df["label"] == label

        ].to_numpy()

        for label in [0, 1]

    }



    rows = []

    assignments = []



    for seed in SEEDS:

        permutations = {

            label: np.random.default_rng(

                seed + label * 10000

            ).permutation(class_patients[label])

            for label in [0, 1]

        }



        query_indices = np.concatenate([

            permutations[0][:3],

            permutations[1][:3],

        ])



        support_pool = {

            label: permutations[label][3:]

            for label in [0, 1]

        }



        for shot in [3, 5]:

            support_indices = np.concatenate([

                support_pool[0][:shot],

                support_pool[1][:shot],

            ])



            support = patient_df.loc[support_indices]

            query = patient_df.loc[query_indices]



            support_x = np.stack(

                support["embedding"].to_list()

            )



            support_y = support["label"].to_numpy(

                dtype=int

            )



            query_x = np.stack(

                query["embedding"].to_list()

            )



            query_y = query["label"].to_numpy(

                dtype=int

            )



            predictions, score = classify(

                support_x,

                support_y,

                query_x,

            )



            metrics = calculate_metrics(

                query_y,

                predictions,

                score,

            )



            rows.append({

                "dataset": "Omni-iEEG",

                "model": "BrainBERT",

                "seed": seed,

                "shot": shot,

                **metrics,

            })



            assignments.append({

                "dataset": "Omni-iEEG",

                "seed": seed,

                "shot": shot,

                "support_patient_ids": "|".join(

                    support["patient_id"].astype(str)

                ),

                "query_patient_ids": "|".join(

                    query["patient_id"].astype(str)

                ),

            })



    results = pd.DataFrame(rows)

    assignments = pd.DataFrame(assignments)



    return results, results.copy(), assignments





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--seeg_manifest", required=True)

    parser.add_argument("--omni_manifest", required=True)

    parser.add_argument("--out_dir", required=True)

    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)



    seeg = pd.read_csv(args.seeg_manifest)

    omni = pd.read_csv(args.omni_manifest)



    seeg_results, seeg_macro, seeg_assignments = (

        within_patient(seeg)

    )



    omni_results, omni_macro, omni_assignments = (

        cross_patient(omni)

    )



    all_results = pd.concat(

        [seeg_macro, omni_macro],

        ignore_index=True,

    )

    metric_columns = [
        "accuracy",
        "balanced_accuracy",

        "precision_sleep",
        "sensitivity_sleep",
        "recall_sleep",
        "f1_sleep",

        "precision_awake",
        "specificity_awake",
        "recall_awake",
        "f1_awake",

        "f1_macro",
        "f1_weighted",

        "mcc",
        "kappa",

        "auroc",
        "auprc",

        "negative_predictive_value",
        "geometric_mean",
        "youden_j",

        "false_positive_rate",
        "false_negative_rate",
    ]




    summary = (

        all_results

        .groupby(

            ["dataset", "model", "shot"],

            as_index=False,

        )[metric_columns]

        .agg(["mean", "std"])

    )



    origin = all_results[

        [

            "dataset",

            "model",

            "seed",

            "shot",

            "auroc",

        ]

    ].copy()



    origin["shot"] = (

        origin["shot"]

        .astype(int)

        .astype(str)

        + "-shot"

    )



    origin = origin.rename(

        columns={"auroc": "AUROC"}

    )



    seeg_results.to_csv(

        out_dir / "seeg_episode_results.csv",

        index=False,

    )



    omni_results.to_csv(

        out_dir / "omni_episode_results.csv",

        index=False,

    )



    all_results.to_csv(

        out_dir / "brainbert_all_seed_results.csv",

        index=False,

    )



    summary.to_csv(

        out_dir / "brainbert_summary.csv",

    )



    origin.to_csv(

        out_dir / "brainbert_origin_auroc.csv",

        index=False,

    )



    seeg_assignments.to_csv(

        out_dir / "seeg_episode_assignments.csv",

        index=False,

    )



    omni_assignments.to_csv(

        out_dir / "omni_episode_assignments.csv",

        index=False,

    )



    print("\nAUROC values for Origin:")

    print(origin.to_string(index=False))



    print("\nSaved in:", out_dir)





if __name__ == "__main__":

    main()

