
import argparse

import json

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

    recall_score,

    roc_auc_score,

)





def normalize_rows(

    array,

    epsilon=1e-12,

):

    norms = np.linalg.norm(

        array,

        axis=1,

        keepdims=True,

    )



    norms = np.maximum(

        norms,

        epsilon,

    )



    return array / norms





def load_embeddings(csv_path):

    dataframe = pd.read_csv(

        csv_path

    )



    embeddings = []



    for path_value in dataframe[

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

                "Expected (192,), found {}."

                .format(embedding.shape)

            )



        embeddings.append(

            embedding

        )



    x = np.stack(

        embeddings,

        axis=0,

    )



    x = normalize_rows(x)



    y = dataframe[

        "label"

    ].astype(int).to_numpy()



    return dataframe, x, y





def stable_softmax(logits):

    logits = logits - np.max(

        logits,

        axis=1,

        keepdims=True,

    )



    exponentials = np.exp(logits)



    return exponentials / np.sum(

        exponentials,

        axis=1,

        keepdims=True,

    )





def calculate_metrics(

    true_labels,

    predicted_labels,

    sleep_scores,

):

    tn, fp, fn, tp = confusion_matrix(

        true_labels,

        predicted_labels,

        labels=[0, 1],

    ).ravel()



    result = {

        "accuracy": float(

            accuracy_score(

                true_labels,

                predicted_labels,

            )

        ),

        "balanced_accuracy": float(

            balanced_accuracy_score(

                true_labels,

                predicted_labels,

            )

        ),

        "precision_sleep": float(

            precision_score(

                true_labels,

                predicted_labels,

                zero_division=0,

            )

        ),

        "sensitivity_sleep": float(

            recall_score(

                true_labels,

                predicted_labels,

                zero_division=0,

            )

        ),

        "specificity_awake": float(

            tn / max(tn + fp, 1)

        ),

        "f1_score": float(

            f1_score(

                true_labels,

                predicted_labels,

                zero_division=0,

            )

        ),

        "mcc": float(

            matthews_corrcoef(

                true_labels,

                predicted_labels,

            )

        ),

        "kappa": float(

            cohen_kappa_score(

                true_labels,

                predicted_labels,

            )

        ),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),

    }



    try:

        result["auroc"] = float(

            roc_auc_score(

                true_labels,

                sleep_scores,

            )

        )

    except ValueError:

        result["auroc"] = float("nan")



    try:

        result["auprc"] = float(

            average_precision_score(

                true_labels,

                sleep_scores,

            )

        )

    except ValueError:

        result["auprc"] = float("nan")



    return result





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--support_csv",

        required=True,

    )



    parser.add_argument(

        "--query_csv",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    parser.add_argument(

        "--shot",

        type=int,

        required=True,

    )



    parser.add_argument(

        "--seed",

        type=int,

        required=True,

    )



    parser.add_argument(

        "--temperature",

        type=float,

        default=1.0,

    )



    args = parser.parse_args()



    if args.temperature <= 0:

        raise ValueError(

            "Temperature must be positive."

        )



    output_dir = Path(

        args.out_dir

    )



    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    (

        support_df,

        support_x,

        support_y,

    ) = load_embeddings(

        args.support_csv

    )



    (

        query_df,

        query_x,

        query_y,

    ) = load_embeddings(

        args.query_csv

    )



    if int((support_y == 0).sum()) != args.shot:

        raise RuntimeError(

            "Incorrect awake support count."

        )



    if int((support_y == 1).sum()) != args.shot:

        raise RuntimeError(

            "Incorrect sleep support count."

        )



    support_patients = set(

        support_df["patient_id"].astype(str)

    )



    query_patients = set(

        query_df["patient_id"].astype(str)

    )



    overlap = (

        support_patients

        .intersection(query_patients)

    )



    if overlap:

        raise RuntimeError(

            "Patient leakage detected: {}"

            .format(sorted(overlap))

        )



    awake_prototype = np.mean(

        support_x[support_y == 0],

        axis=0,

        keepdims=True,

    )



    sleep_prototype = np.mean(

        support_x[support_y == 1],

        axis=0,

        keepdims=True,

    )



    awake_prototype = normalize_rows(

        awake_prototype

    )[0]



    sleep_prototype = normalize_rows(

        sleep_prototype

    )[0]



    prototypes = np.stack(

        [

            awake_prototype,

            sleep_prototype,

        ],

        axis=0,

    )



    cosine_similarities = np.matmul(

        query_x,

        prototypes.T,

    )



    predicted_labels = np.argmax(

        cosine_similarities,

        axis=1,

    ).astype(int)



    probabilities = stable_softmax(

        cosine_similarities

        / args.temperature

    )



    sleep_probabilities = probabilities[

        :,

        1

    ]



    metrics = calculate_metrics(

        query_y,

        predicted_labels,

        sleep_probabilities,

    )



    result = {

        "seed": int(args.seed),

        "shot": int(args.shot),

        "support_patients": int(

            len(support_df)

        ),

        "query_patients": int(

            len(query_df)

        ),

        "temperature": float(

            args.temperature

        ),

    }



    result.update(metrics)



    pd.DataFrame(

        [result]

    ).to_csv(

        output_dir / "metrics.csv",

        index=False,

    )



    predictions = query_df.copy()



    predictions["true_label"] = query_y

    predictions["predicted_label"] = (

        predicted_labels

    )



    predictions["true_state"] = np.where(

        query_y == 1,

        "sleep",

        "awake",

    )



    predictions["predicted_state"] = np.where(

        predicted_labels == 1,

        "sleep",

        "awake",

    )



    predictions["similarity_awake"] = (

        cosine_similarities[:, 0]

    )



    predictions["similarity_sleep"] = (

        cosine_similarities[:, 1]

    )



    predictions["probability_sleep"] = (

        sleep_probabilities

    )



    predictions.to_csv(

        output_dir / "query_predictions.csv",

        index=False,

    )



    np.savez(

        output_dir / "class_prototypes.npz",

        awake_prototype=awake_prototype,

        sleep_prototype=sleep_prototype,

    )



    with open(

        output_dir / "metrics.json",

        "w",

    ) as file:

        json.dump(

            result,

            file,

            indent=2,

        )



    print("\nSeed:", args.seed)

    print("Shot:", args.shot)



    print("\nQuery metrics:")



    for key, value in metrics.items():

        print(

            "{}: {}".format(

                key,

                value,

            )

        )



    print("\nSaved:", output_dir)





if __name__ == "__main__":

    main()

