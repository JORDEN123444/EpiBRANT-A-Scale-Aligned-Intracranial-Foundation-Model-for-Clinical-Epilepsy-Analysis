
import argparse

import inspect

from pathlib import Path



import numpy as np

import pandas as pd

import torch

import torch.nn as nn

import yaml

import matplotlib.pyplot as plt



from torch.utils.data import DataLoader

from sklearn.manifold import TSNE

from sklearn.preprocessing import StandardScaler



from downstream.seizure_detection.data_seizure import SeizureWindowDataset

import downstream.seizure_detection.model_seizure as model_module





def find_checkpoint(run_dir):

    run_dir = Path(run_dir)



    for name in ["best_model.pt", "best.pt", "checkpoint_best.pt", "model_best.pt", "last_model.pt"]:

        p = run_dir / name

        if p.exists():

            return p



    pts = sorted(run_dir.glob("*.pt"))

    if len(pts) == 0:

        raise FileNotFoundError(f"No checkpoint found in {run_dir}")



    return pts[0]





def get_state_dict(ckpt):

    if isinstance(ckpt, dict):

        for key in ["model", "model_state_dict", "state_dict"]:

            if key in ckpt:

                return ckpt[key]

    return ckpt





def build_model(cfg):

    model_cfg = dict(cfg["model"])

    data_cfg = cfg["data"]



    seq_len = int(float(data_cfg["context_sec"]) / float(data_cfg["patch_sec"]))

    max_channels = int(data_cfg["max_channels"])



    candidate_names = [

        "SABrantSeizureClassifier",

        "SeizureClassifier",

        "SeizureDetectionModel",

        "SeizureModel",

        "SA_Brant_SeizureClassifier",

    ]



    for name in candidate_names:

        if not hasattr(model_module, name):

            continue



        cls = getattr(model_module, name)



        trials = [

            dict(**model_cfg, seq_len=seq_len, max_channels=max_channels),

            dict(**model_cfg, seq_len=seq_len, max_channels=max_channels, num_classes=2),

            dict(**model_cfg, max_channels=max_channels),

            dict(**model_cfg),

        ]



        for kwargs in trials:

            try:

                model = cls(**kwargs)

                print(f"Built model using class: {name}")

                print("Model args:", list(kwargs.keys()))

                return model

            except TypeError:

                pass



    print("\nCould not automatically build model.")

    print("Available classes in model_seizure.py:")

    for n, obj in inspect.getmembers(model_module):

        if inspect.isclass(obj):

            print("  ", n)



    raise RuntimeError("Model class not found. Send me: grep -n \"class \" downstream/seizure_detection/model_seizure.py")





def register_penultimate_hook(model):

    linear_layers = []



    for name, module in model.named_modules():

        if isinstance(module, nn.Linear):

            linear_layers.append((name, module))



    if len(linear_layers) == 0:

        raise RuntimeError("No Linear layer found. Cannot capture penultimate features.")



    layer_name, layer = linear_layers[-1]

    print("Capturing representation before final Linear layer:", layer_name)



    cache = {}



    def hook(module, inp, out):

        z = inp[0].detach()

        if z.ndim > 2:

            z = z.reshape(z.shape[0], -1)

        cache["feat"] = z.cpu()



    handle = layer.register_forward_hook(hook)

    return cache, handle





@torch.no_grad()

def collect_embeddings(model, dataset, device, max_points):

    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)



    cache, handle = register_penultimate_hook(model)



    feats = []

    labels = []

    patients = []

    files = []



    model.eval()



    for i, batch in enumerate(loader):

        if i >= max_points:

            break



        x = batch["x"].to(device).float()



        try:

            fs = batch["fs"].to(device)

        except Exception:

            fs = None



        try:

            channel_mask = batch["channel_mask"].to(device).float()

        except Exception:

            channel_mask = None



        try:

            if fs is not None and channel_mask is not None:

                _ = model(x, fs, channel_mask)

            else:

                _ = model(x)

        except TypeError:

            _ = model(x)



        if "feat" not in cache:

            raise RuntimeError("Feature hook did not capture embeddings.")



        feats.append(cache["feat"].numpy()[0])



        y = batch["label"]

        if torch.is_tensor(y):

            y = int(y.item())

        else:

            y = int(y[0])

        labels.append(y)



        if "patient" in batch:

            patients.append(str(batch["patient"][0]))

        elif "subject" in batch:

            patients.append(str(batch["subject"][0]))

        else:

            patients.append("unknown")



        if "original_name" in batch:

            files.append(str(batch["original_name"][0]))

        else:

            files.append("unknown")



    handle.remove()



    return np.asarray(feats), np.asarray(labels), patients, files





def plot_tsne(df, out_dir, split):

    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.size": 8,

        "axes.labelsize": 8,

        "xtick.labelsize": 7,

        "ytick.labelsize": 7,

        "legend.fontsize": 7,

        "axes.linewidth": 0.8,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })



    fig, ax = plt.subplots(figsize=(4.8, 4.2))



    for label, name, marker in [(0, "Normal", "o"), (1, "Seizure", "^")]:

        d = df[df["label"] == label]

        ax.scatter(

            d["tsne_1"],

            d["tsne_2"],

            s=18,

            alpha=0.78,

            marker=marker,

            label=f"{name} (n={len(d)})",

            edgecolors="none",

        )



    ax.set_xlabel("t-SNE 1")

    ax.set_ylabel("t-SNE 2")

    ax.legend(frameon=False)

    ax.set_title(f"Fine-tuned embeddings: {split} split")

    fig.tight_layout()



    base = out_dir / f"fig_tsne_{split}_seizure_vs_normal"

    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")

    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")

    fig.savefig(str(base) + ".pdf", dpi=600, bbox_inches="tight")

    plt.close(fig)



    print("Saved:", str(base) + ".png")

    print("Saved:", str(base) + ".tiff")

    print("Saved:", str(base) + ".pdf")





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument("--run_dir", required=True)

    parser.add_argument("--split", choices=["train", "val", "test"], default="test")

    parser.add_argument("--max_points", type=int, default=800)

    parser.add_argument("--perplexity", type=float, default=30.0)

    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()



    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)



    out_dir = Path(args.run_dir) / "tsne_finetune_figures"

    out_dir.mkdir(parents=True, exist_ok=True)



    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Device:", device)



    model = build_model(cfg).to(device)



    ckpt_path = find_checkpoint(args.run_dir)

    print("Loading checkpoint:", ckpt_path)



    ckpt = torch.load(ckpt_path, map_location=device)

    state = get_state_dict(ckpt)



    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing keys:", len(missing))

    print("Unexpected keys:", len(unexpected))



    dataset = SeizureWindowDataset(

        manifest_csv=cfg["data"]["manifest_csv"],

        split=args.split,

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

    )



    print("Split:", args.split)

    print("Dataset size:", len(dataset))



    feats, labels, patients, files = collect_embeddings(

        model=model,

        dataset=dataset,

        device=device,

        max_points=args.max_points,

    )



    print("Collected embeddings:", feats.shape)

    print("Class counts:", pd.Series(labels).value_counts().to_dict())



    feats = StandardScaler().fit_transform(feats)



    n = feats.shape[0]

    perplexity = min(args.perplexity, max(5, (n - 1) // 3))

    print("t-SNE perplexity:", perplexity)



    tsne = TSNE(

        n_components=2,

        perplexity=perplexity,

        init="pca",

        learning_rate="auto",

        random_state=args.seed,

    )



    z = tsne.fit_transform(feats)



    df = pd.DataFrame({

        "tsne_1": z[:, 0],

        "tsne_2": z[:, 1],

        "label": labels,

        "class_name": ["Normal" if y == 0 else "Seizure" for y in labels],

        "patient": patients,

        "file": files,

        "split": args.split,

    })



    csv_path = out_dir / f"tsne_embeddings_{args.split}.csv"

    df.to_csv(csv_path, index=False)

    print("Saved:", csv_path)



    plot_tsne(df, out_dir, args.split)





if __name__ == "__main__":

    main()

