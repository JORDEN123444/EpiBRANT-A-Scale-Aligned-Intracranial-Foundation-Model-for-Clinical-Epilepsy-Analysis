
import argparse

import inspect

from pathlib import Path



import matplotlib

matplotlib.use("Agg")



import matplotlib.pyplot as plt

import numpy as np

import pandas as pd

from sklearn.decomposition import PCA

from sklearn.manifold import TSNE

from sklearn.preprocessing import normalize





COLORS = {

    "awake": "#0072B2",

    "sleep": "#D55E00",

}





def calculate_coordinates(

    manifest_path,

    dataset_name,

    seed,

):

    df = pd.read_csv(

        manifest_path

    )



    embeddings = np.stack([

        np.load(

            path,

            allow_pickle=False,

        )

        for path in df["embedding_path"]

    ])



    embeddings = normalize(

        embeddings,

        norm="l2",

        axis=1,

    )



    components = min(

        50,

        embeddings.shape[1],

        embeddings.shape[0] - 1,

    )



    reduced = PCA(

        n_components=components,

        random_state=seed,

    ).fit_transform(

        embeddings

    )



    perplexity = min(

        30.0,

        (len(df) - 1) / 3.0,

    )



    kwargs = {

        "n_components": 2,

        "perplexity": max(

            2.0,

            perplexity,

        ),

        "learning_rate": 200.0,

        "init": "pca",

        "random_state": seed,

        "metric": "euclidean",

        "method": "barnes_hut",

        "angle": 0.5,

    }



    parameters = inspect.signature(

        TSNE

    ).parameters



    if "max_iter" in parameters:

        kwargs["max_iter"] = 2000

    else:

        kwargs["n_iter"] = 2000



    coordinates = TSNE(

        **kwargs

    ).fit_transform(

        reduced

    )



    output = df.copy()

    output["tsne_1"] = coordinates[:, 0]

    output["tsne_2"] = coordinates[:, 1]

    output["display_dataset"] = dataset_name

    output["tsne_seed"] = seed

    output["tsne_perplexity"] = kwargs[

        "perplexity"

    ]



    return output





def draw_panel(

    axis,

    df,

    title,

    letter,

    point_size,

):

    class_order = (

        df["state"]

        .value_counts()

        .sort_values(

            ascending=False

        )

        .index

    )



    handles = {}



    for state in class_order:

        subset = df[

            df["state"] == state

        ]



        handles[state] = axis.scatter(

            subset["tsne_1"],

            subset["tsne_2"],

            s=point_size,

            c=COLORS[state],

            alpha=0.65,

            edgecolors="none",

            linewidths=0,

            rasterized=True,

            label=state.capitalize(),

        )



    axis.set_xticks([])

    axis.set_yticks([])

    axis.set_xlabel("")

    axis.set_ylabel("")

    axis.set_title(

        title,

        fontsize=8,

        pad=3,

    )



    for spine in axis.spines.values():

        spine.set_visible(False)



    axis.text(

        -0.04,

        1.02,

        letter,

        transform=axis.transAxes,

        fontsize=9,

        fontweight="bold",

        ha="right",

        va="bottom",

    )



    return handles





def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--seeg_manifest",

        required=True,

    )



    parser.add_argument(

        "--omni_manifest",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    parser.add_argument(

        "--point_size",

        type=float,

        default=7.0,

    )



    args = parser.parse_args()



    out_dir = Path(

        args.out_dir

    )



    out_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    seeg = calculate_coordinates(

        args.seeg_manifest,

        "SEEG dataset",

        seed=42,

    )



    omni = calculate_coordinates(

        args.omni_manifest,

        "Omni-iEEG dataset",

        seed=43,

    )



    seeg.to_csv(

        out_dir

        / "seeg_brainbert_tsne_coordinates.csv",

        index=False,

    )



    omni.to_csv(

        out_dir

        / "omni_ieeg_brainbert_tsne_coordinates.csv",

        index=False,

    )



    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.sans-serif": [

            "Arial",

            "Helvetica",

            "DejaVu Sans",

        ],

        "font.size": 7,

        "axes.titlesize": 8,

        "legend.fontsize": 7,

        "figure.facecolor": "white",

        "axes.facecolor": "white",

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

        "svg.fonttype": "none",

    })



    figure, axes = plt.subplots(

        1,

        2,

        figsize=(

            180 / 25.4,

            72 / 25.4,

        ),

    )



    handles = draw_panel(

        axes[0],

        seeg,

        "SEEG dataset",

        "a",

        args.point_size,

    )



    draw_panel(

        axes[1],

        omni,

        "Omni-iEEG dataset",

        "b",

        args.point_size,

    )



    figure.legend(

        handles=[

            handles["awake"],

            handles["sleep"],

        ],

        labels=[

            "Awake",

            "Sleep",

        ],

        loc="upper center",

        bbox_to_anchor=(

            0.5,

            0.995,

        ),

        ncol=2,

        frameon=False,

        handletextpad=0.3,

        columnspacing=1.4,

        markerscale=1.3,

    )



    figure.subplots_adjust(

        left=0.02,

        right=0.995,

        bottom=0.03,

        top=0.86,

        wspace=0.08,

    )



    base = (

        out_dir

        / "brainbert_all_segments_tsne_npj"

    )



    figure.savefig(

        str(base) + ".png",

        dpi=600,

        bbox_inches="tight",

        pad_inches=0.015,

    )



    figure.savefig(

        str(base) + ".tiff",

        dpi=600,

        bbox_inches="tight",

        pad_inches=0.015,

        pil_kwargs={

            "compression": "tiff_lzw",

        },

    )



    figure.savefig(

        str(base) + ".pdf",

        bbox_inches="tight",

        pad_inches=0.015,

    )



    figure.savefig(

        str(base) + ".svg",

        bbox_inches="tight",

        pad_inches=0.015,

    )



    plt.close(

        figure

    )



    print("SEEG points:", len(seeg))

    print(seeg["state"].value_counts())



    print("\nOmni-iEEG points:", len(omni))

    print(omni["state"].value_counts())



    print("\nSaved:")

    print(str(base) + ".png")

    print(str(base) + ".tiff")

    print(str(base) + ".pdf")

    print(str(base) + ".svg")





if __name__ == "__main__":

    main()

