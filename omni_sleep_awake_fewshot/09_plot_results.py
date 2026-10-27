
import argparse

from pathlib import Path



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--results",

        required=True,

    )



    parser.add_argument(

        "--summary",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    args = parser.parse_args()



    results = pd.read_csv(

        args.results

    )



    summary = pd.read_csv(

        args.summary

    )



    output_dir = Path(

        args.out_dir

    )



    output_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    metrics = [

        ("balanced_accuracy", "Balanced accuracy"),

        ("f1_score", "F1 score"),

        ("mcc", "MCC"),

        ("auroc", "AUROC"),

        ("auprc", "AUPRC"),

    ]



    shots = [3, 5]



    colors = {

        3: "#4C78A8",

        5: "#F28E2B",

    }



    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": [

            "Arial",

            "DejaVu Sans",

        ],

        "font.size": 8,

        "axes.labelsize": 9,

        "xtick.labelsize": 8,

        "ytick.labelsize": 8,

        "legend.fontsize": 8,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })



    x = np.arange(

        len(metrics)

    )



    width = 0.34



    offsets = {

        3: -width / 2,

        5: width / 2,

    }



    fig, ax = plt.subplots(

        figsize=(4.5, 3.0)

    )



    rng = np.random.default_rng(42)



    minimum_value = 0.0



    for shot in shots:

        means = []

        standard_deviations = []



        for metric, _ in metrics:

            row = summary[

                (

                    summary["shot"].astype(int)

                    == shot

                )

                & (

                    summary["metric"] == metric

                )

            ]



            if len(row) != 1:

                raise RuntimeError(

                    "Missing summary row for "

                    "{}-shot {}."

                    .format(shot, metric)

                )



            means.append(

                float(row.iloc[0]["mean"])

            )



            standard_deviations.append(

                float(

                    row.iloc[0][

                        "standard_deviation"

                    ]

                )

            )



        positions = (

            x + offsets[shot]

        )



        ax.bar(

            positions,

            means,

            width=width,

            yerr=standard_deviations,

            capsize=2.5,

            color=colors[shot],

            edgecolor=colors[shot],

            linewidth=0.8,

            error_kw={

                "elinewidth": 0.8,

                "capthick": 0.8,

            },

            label="{}-shot".format(shot),

            zorder=2,

        )



        shot_results = results[

            results["shot"].astype(int)

            == shot

        ]



        for metric_index, (

            metric,

            _,

        ) in enumerate(metrics):

            values = pd.to_numeric(

                shot_results[metric],

                errors="coerce",

            ).dropna().to_numpy()



            if len(values) > 0:

                minimum_value = min(

                    minimum_value,

                    float(values.min()),

                )



            jitter = rng.uniform(

                -0.035,

                0.035,

                size=len(values),

            )



            ax.scatter(

                np.full(

                    len(values),

                    positions[metric_index],

                ) + jitter,

                values,

                s=18,

                facecolors="white",

                edgecolors="#4d4d4d",

                linewidths=0.7,

                zorder=4,

            )



    ax.set_xticks(x)



    ax.set_xticklabels(

        [

            label

            for _, label in metrics

        ],

        rotation=30,

        ha="right",

    )



    lower_limit = (

        -0.2

        if minimum_value < 0

        else 0.0

    )



    ax.set_ylim(

        lower_limit,

        1.08,

    )



    ax.set_ylabel(

        "Patient-level performance"

    )



    ax.spines["top"].set_visible(

        False

    )



    ax.spines["right"].set_visible(

        False

    )



    ax.yaxis.grid(

        True,

        linewidth=0.4,

        alpha=0.25,

        zorder=1,

    )



    ax.set_axisbelow(True)



    ax.legend(

        frameon=False,

        loc="upper center",

        bbox_to_anchor=(0.5, 1.08),

        ncol=2,

    )



    plt.tight_layout()



    png_path = (

        output_dir

        / "omni_sleep_awake_3shot_vs_5shot.png"

    )



    tiff_path = (

        output_dir

        / "omni_sleep_awake_3shot_vs_5shot.tiff"

    )



    pdf_path = (

        output_dir

        / "omni_sleep_awake_3shot_vs_5shot.pdf"

    )



    fig.savefig(

        png_path,

        dpi=600,

        bbox_inches="tight",

    )



    fig.savefig(

        tiff_path,

        dpi=600,

        bbox_inches="tight",

    )



    fig.savefig(

        pdf_path,

        bbox_inches="tight",

    )



    print("Saved:")

    print(png_path)

    print(tiff_path)

    print(pdf_path)



    plt.show()





if __name__ == "__main__":

    main()

