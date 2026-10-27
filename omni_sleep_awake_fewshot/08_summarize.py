
import argparse

from pathlib import Path



import pandas as pd





METRICS = [

    "accuracy",

    "balanced_accuracy",

    "precision_sleep",

    "sensitivity_sleep",

    "specificity_awake",

    "f1_score",

    "mcc",

    "kappa",

    "auroc",

    "auprc",

]





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--episode_table",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    args = parser.parse_args()



    episodes = pd.read_csv(

        args.episode_table

    )



    output_dir = Path(

        args.out_dir

    )



    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    rows = []

    missing_results = []



    for _, episode in episodes.iterrows():

        metrics_path = (

            Path(str(episode["result_dir"]))

            / "metrics.csv"

        )



        if not metrics_path.exists():

            missing_results.append(

                str(metrics_path)

            )

            continue



        metrics = pd.read_csv(

            metrics_path

        )



        if len(metrics) != 1:

            raise RuntimeError(

                "Expected one row in {}."

                .format(metrics_path)

            )



        row = metrics.iloc[0].to_dict()



        row["support_patient_ids"] = (

            episode["support_patient_ids"]

        )



        row["query_patient_ids"] = (

            episode["query_patient_ids"]

        )



        row["metrics_path"] = str(

            metrics_path

        )



        rows.append(row)



    if missing_results:

        print("Missing result files:")



        for path in missing_results:

            print(path)



        raise RuntimeError(

            "{} experiments are incomplete."

            .format(len(missing_results))

        )



    results = pd.DataFrame(

        rows

    ).sort_values(

        ["shot", "seed"]

    )



    all_results_path = (

        output_dir

        / "omni_all_episode_results.csv"

    )



    results.to_csv(

        all_results_path,

        index=False,

    )



    summary_rows = []



    for shot, group in results.groupby(

        "shot"

    ):

        for metric in METRICS:

            values = pd.to_numeric(

                group[metric],

                errors="coerce",

            )



            summary_rows.append({

                "shot": int(shot),

                "metric": metric,

                "number_of_seeds": int(

                    values.notna().sum()

                ),

                "mean": float(

                    values.mean()

                ),

                "standard_deviation": float(

                    values.std(ddof=1)

                ),

                "median": float(

                    values.median()

                ),

                "minimum": float(

                    values.min()

                ),

                "maximum": float(

                    values.max()

                ),

            })



    summary = pd.DataFrame(

        summary_rows

    )



    summary_path = (

        output_dir

        / "omni_fewshot_summary.csv"

    )



    summary.to_csv(

        summary_path,

        index=False,

    )



    paired_rows = []



    for metric in METRICS:

        pivot = results.pivot(

            index="seed",

            columns="shot",

            values=metric,

        )



        if 3 not in pivot.columns:

            continue



        if 5 not in pivot.columns:

            continue



        for seed in pivot.index:

            paired_rows.append({

                "seed": int(seed),

                "metric": metric,

                "three_shot": float(

                    pivot.loc[seed, 3]

                ),

                "five_shot": float(

                    pivot.loc[seed, 5]

                ),

                "five_minus_three": float(

                    pivot.loc[seed, 5]

                    - pivot.loc[seed, 3]

                ),

            })



    paired = pd.DataFrame(

        paired_rows

    )



    paired_path = (

        output_dir

        / "omni_paired_3shot_vs_5shot.csv"

    )



    paired.to_csv(

        paired_path,

        index=False,

    )



    print("\nOMNI-iEEG PATIENT-LEVEL RESULTS")

    print("=" * 80)



    for shot in sorted(

        summary["shot"].unique()

    ):

        print(

            "\n{}-shot".format(

                int(shot)

            )

        )



        shot_summary = summary[

            summary["shot"] == shot

        ]



        for metric in METRICS:

            row = shot_summary[

                shot_summary["metric"]

                == metric

            ].iloc[0]



            print(

                "{:<22s}: {:.4f} ± {:.4f}"

                .format(

                    metric,

                    row["mean"],

                    row["standard_deviation"],

                )

            )



    print("\nPaired 5-shot minus 3-shot changes:")



    for metric in METRICS:

        values = paired[

            paired["metric"] == metric

        ]["five_minus_three"]



        print(

            "{:<22s}: {:+.4f} ± {:.4f}"

            .format(

                metric,

                values.mean(),

                values.std(ddof=1),

            )

        )



    print("\nSaved:")

    print(all_results_path)

    print(summary_path)

    print(paired_path)





if __name__ == "__main__":

    main()

