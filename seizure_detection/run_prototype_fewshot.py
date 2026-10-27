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
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    fbeta_score,
    cohen_kappa_score,
    matthews_corrcoef,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from downstream.seizure_detection.data_seizure import SeizureWindowDataset
import downstream.seizure_detection.model_seizure as model_module


# =========================================================
# Basic helpers
# =========================================================
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
        for key in ["model", "model_state_dict", "state_dict", "net", "network"]:
            if key in ckpt:
                return ckpt[key]
    return ckpt


def build_model(cfg):
    model_cfg = dict(cfg["model"])
    data_cfg = cfg["data"]

    seq_len = int(float(data_cfg["context_sec"]) / float(data_cfg["patch_sec"]))
    max_channels = int(data_cfg["max_channels"])

    base_kwargs = dict(model_cfg)
    base_kwargs["seq_len"] = seq_len
    base_kwargs["max_channels"] = max_channels

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

        trials = []
        trials.append(dict(base_kwargs))
        tmp = dict(base_kwargs); tmp["num_classes"] = 2; trials.append(tmp)
        tmp = dict(base_kwargs); tmp["n_classes"] = 2; trials.append(tmp)
        tmp = dict(base_kwargs); tmp["n_outputs"] = 2; trials.append(tmp)
        trials.append(dict(model_cfg))
        tmp = dict(model_cfg); tmp["num_classes"] = 2; trials.append(tmp)

        for kwargs in trials:
            try:
                sig = inspect.signature(cls.__init__)
                params = sig.parameters
                has_var_kw = any(p.kind == p.VAR_KEYWORD for p in params.values())

                if has_var_kw:
                    filtered = kwargs
                else:
                    allowed = set(params.keys()) - {"self"}
                    filtered = {k: v for k, v in kwargs.items() if k in allowed}

                model = cls(**filtered)
                print(f"Built model using class: {name}")
                print("Model args:", filtered)
                return model
            except Exception:
                continue

    print("\nCould not automatically build model.")
    print("Available classes in model_seizure.py:")
    for n, obj in inspect.getmembers(model_module):
        if inspect.isclass(obj):
            print("  ", n)

    raise RuntimeError("Model class/init failed. Send me: grep -n \"class SABrantSeizureClassifier\\|def __init__\" downstream/seizure_detection/model_seizure.py")


def register_penultimate_hook(model):
    linear_layers = []

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module))

    if len(linear_layers) == 0:
        raise RuntimeError("No Linear layer found. Cannot capture penultimate embeddings.")

    layer_name, layer = linear_layers[-1]
    print("Capturing embedding before final Linear layer:", layer_name)

    cache = {}

    def hook(module, inp, out):
        z = inp[0].detach()
        if z.ndim > 2:
            z = z.reshape(z.shape[0], -1)
        cache["feat"] = z.cpu()

    handle = layer.register_forward_hook(hook)
    return cache, handle


def get_batch_x_y(batch):
    if isinstance(batch, dict):
        x = batch["x"]
        if "label" in batch:
            y = batch["label"]
        elif "y" in batch:
            y = batch["y"]
        else:
            raise RuntimeError("Cannot find label in batch dictionary.")

        fs = batch.get("fs", None)
        channel_mask = batch.get("channel_mask", None)
        return x, y, fs, channel_mask

    if isinstance(batch, (list, tuple)):
        x = batch[0]
        y = batch[1]
        fs = None
        channel_mask = None
        return x, y, fs, channel_mask

    raise RuntimeError("Unknown batch format.")


@torch.no_grad()
def extract_embeddings(model, dataset, device, batch_size=8, max_points=None):
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    cache, handle = register_penultimate_hook(model)

    all_feat = []
    all_label = []

    model.eval()

    seen = 0

    for batch in loader:
        x, y, fs, channel_mask = get_batch_x_y(batch)

        x = x.to(device).float()

        try:
            fs = fs.to(device) if fs is not None else None
        except Exception:
            fs = None

        try:
            channel_mask = channel_mask.to(device).float() if channel_mask is not None else None
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
            raise RuntimeError("Embedding hook did not capture features.")

        feat = cache["feat"].numpy()
        lab = y.detach().cpu().numpy().astype(int) if torch.is_tensor(y) else np.asarray(y).astype(int)

        all_feat.append(feat)
        all_label.append(lab)

        seen += len(lab)
        if max_points is not None and seen >= max_points:
            break

    handle.remove()

    X = np.concatenate(all_feat, axis=0)
    y = np.concatenate(all_label, axis=0)

    if max_points is not None:
        X = X[:max_points]
        y = y[:max_points]

    return X, y


# =========================================================
# Prototype few-shot functions
# =========================================================
def make_support_indices(y_train, shot, rng):
    idx0 = np.where(y_train == 0)[0]
    idx1 = np.where(y_train == 1)[0]

    if len(idx0) < shot or len(idx1) < shot:
        raise RuntimeError(f"Not enough support examples for {shot}-shot. Normal={len(idx0)}, Seizure={len(idx1)}")

    support0 = rng.choice(idx0, size=shot, replace=False)
    support1 = rng.choice(idx1, size=shot, replace=False)

    return np.concatenate([support0, support1])


def compute_prototypes(X_support, y_support):
    proto0 = X_support[y_support == 0].mean(axis=0)
    proto1 = X_support[y_support == 1].mean(axis=0)
    return np.stack([proto0, proto1], axis=0)


def prototype_scores(X, prototypes):
    d0 = np.linalg.norm(X - prototypes[0][None, :], axis=1)
    d1 = np.linalg.norm(X - prototypes[1][None, :], axis=1)

    # Probability-like seizure score from negative distances
    a0 = -d0
    a1 = -d1
    m = np.maximum(a0, a1)
    e0 = np.exp(a0 - m)
    e1 = np.exp(a1 - m)
    p_seizure = e1 / (e0 + e1 + 1e-12)

    # Direct prediction by nearest prototype
    pred_nearest = (d1 < d0).astype(int)

    return p_seizure, pred_nearest, d0, d1


def metrics_from_scores(y, score, threshold):
    pred = (score >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()

    out = {
        "accuracy": accuracy_score(y, pred),
        "balanced_accuracy": balanced_accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "sensitivity": recall_score(y, pred, zero_division=0),
        "specificity": tn / max(1, tn + fp),
        "f1_score": f1_score(y, pred, zero_division=0),
        "f2_score": fbeta_score(y, pred, beta=2, zero_division=0),
        "kappa": cohen_kappa_score(y, pred),
        "mcc": matthews_corrcoef(y, pred),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

    try:
        out["auroc"] = roc_auc_score(y, score)
    except Exception:
        out["auroc"] = np.nan

    try:
        out["auprc"] = average_precision_score(y, score)
    except Exception:
        out["auprc"] = np.nan

    return out


def select_threshold_by_validation(y_val, score_val):
    thresholds = np.linspace(0.0, 1.0, 201)

    best_th = 0.5
    best_ba = -1

    for th in thresholds:
        m = metrics_from_scores(y_val, score_val, th)
        ba = m["balanced_accuracy"]

        if ba > best_ba:
            best_ba = ba
            best_th = th

    return best_th


def summarize_mean_std(df):
    metric_cols = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "f2_score",
        "kappa",
        "mcc",
        "auroc",
        "auprc",
        "threshold",
    ]

    rows = []

    for (split, shot), g in df.groupby(["split", "shot"]):
        row = {
            "split": split,
            "shot": shot,
            "n_runs": len(g),
        }

        for m in metric_cols:
            row[m + "_mean"] = g[m].mean()
            row[m + "_std"] = g[m].std(ddof=1)

        rows.append(row)

    return pd.DataFrame(rows).sort_values(["split", "shot"])


# =========================================================
# Figure functions
# =========================================================
def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "axes.linewidth": 0.8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_all(fig, out_base):
    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(str(out_base) + ".tiff", dpi=600, bbox_inches="tight")
    fig.savefig(str(out_base) + ".pdf", dpi=600, bbox_inches="tight")
    print("Saved:", str(out_base) + ".png")
    print("Saved:", str(out_base) + ".tiff")
    print("Saved:", str(out_base) + ".pdf")


def plot_metric_small_multiples(summary, out_dir, split="test"):
    set_style()

    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("f1_score", "F1-score"),
        ("precision", "Precision"),
        ("sensitivity", "Sensitivity"),
        ("specificity", "Specificity"),
        ("auroc", "AUROC"),
        ("auprc", "AUPRC"),
        ("mcc", "MCC"),
    ]

    d = summary[summary["split"] == split].copy()

    fig, axes = plt.subplots(2, 4, figsize=(8.8, 4.6))
    axes = axes.ravel()

    for ax, (m, title) in zip(axes, metrics):
        shots = d["shot"].values
        x = np.arange(len(shots))
        means = d[m + "_mean"].values
        stds = d[m + "_std"].values

        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, linewidth=1.3, markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s)}-shot" for s in shots])
        ax.set_title(title)
        ax.set_ylim(0, 1.05 if m not in ["mcc"] else 1.0)

        if m == "mcc":
            ax.set_ylim(-1.0, 1.0)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Prototype few-shot classification: {split} performance", y=1.02)
    fig.tight_layout()
    save_all(fig, out_dir / f"fig_metrics_mean_std_{split}_3shot_4shot_no_legend")
    plt.close(fig)


def plot_auc_only(summary, out_dir, split="test"):
    set_style()

    d = summary[summary["split"] == split].copy()

    fig, axes = plt.subplots(1, 2, figsize=(5.2, 3.0))

    for ax, m, title in [
        (axes[0], "auroc", "AUROC"),
        (axes[1], "auprc", "AUPRC"),
    ]:
        shots = d["shot"].values
        x = np.arange(len(shots))
        means = d[m + "_mean"].values
        stds = d[m + "_std"].values

        ax.errorbar(x, means, yerr=stds, fmt="o-", capsize=3, linewidth=1.4, markersize=4)
        ax.set_xticks(x)
        ax.set_xticklabels([f"{int(s)}-shot" for s in shots])
        ax.set_ylim(0, 1.05)
        ax.set_title(title)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.tight_layout()
    save_all(fig, out_dir / f"fig_auc_mean_std_{split}_3shot_4shot_no_legend")
    plt.close(fig)


def plot_tsne_two_shots(X_query, y_query, X_support_by_shot, y_support_by_shot, out_dir, split, seed):
    set_style()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))

    for ax, shot in zip(axes, sorted(X_support_by_shot.keys())):
        X_support = X_support_by_shot[shot]
        y_support = y_support_by_shot[shot]

        X_all = np.concatenate([X_query, X_support], axis=0)
        y_all = np.concatenate([y_query, y_support], axis=0)
        source = np.array(["query"] * len(y_query) + ["support"] * len(y_support))

        X_scaled = StandardScaler().fit_transform(X_all)

        n = X_scaled.shape[0]
        perplexity = min(30, max(5, (n - 1) // 3))

        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate=200.0,
            random_state=seed,
            n_iter=1000,
        )

        Z = tsne.fit_transform(X_scaled)

        # Query points
        q0 = (source == "query") & (y_all == 0)
        q1 = (source == "query") & (y_all == 1)

        ax.scatter(Z[q0, 0], Z[q0, 1], s=12, alpha=0.55, marker="o", edgecolors="none")
        ax.scatter(Z[q1, 0], Z[q1, 1], s=14, alpha=0.75, marker="^", edgecolors="none")

        # Support points
        s0 = (source == "support") & (y_all == 0)
        s1 = (source == "support") & (y_all == 1)

        ax.scatter(Z[s0, 0], Z[s0, 1], s=55, marker="o", edgecolors="black", linewidths=0.8)
        ax.scatter(Z[s1, 0], Z[s1, 1], s=65, marker="^", edgecolors="black", linewidths=0.8)

        ax.set_title(f"{shot}-shot")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(f"Prototype few-shot embeddings: {split} split", y=1.02)
    fig.tight_layout()

    save_all(fig, out_dir / f"fig_tsne_{split}_3shot_4shot_seed{seed}_no_legend")
    plt.close(fig)


# =========================================================
# Main
# =========================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--checkpoint_path", default=None, help="Optional path to a specific .pt checkpoint")
    parser.add_argument("--out_name", default="prototype_fewshot_results", help="Output folder name inside run_dir")
    parser.add_argument("--shots", nargs="+", type=int, default=[3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    parser.add_argument("--support_split", default="train")
    parser.add_argument("--val_split", default="val")
    parser.add_argument("--test_split", default="test")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_query_points_tsne", type=int, default=800)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    run_dir = Path(args.run_dir)
    out_dir = run_dir / args.out_name
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    model = build_model(cfg).to(device)

    ckpt_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else find_checkpoint(run_dir)
    print("Loading checkpoint:", ckpt_path)

    ckpt = torch.load(ckpt_path, map_location=device)
    state = get_state_dict(ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)

    print("Missing keys:", len(missing))
    print("Unexpected keys:", len(unexpected))

    print("\nLoading datasets...")

    ds_train = SeizureWindowDataset(
        manifest_csv=cfg["data"]["manifest_csv"],
        split=args.support_split,
        context_sec=cfg["data"]["context_sec"],
        patch_sec=cfg["data"]["patch_sec"],
        max_channels=cfg["data"]["max_channels"],
    )

    ds_val = SeizureWindowDataset(
        manifest_csv=cfg["data"]["manifest_csv"],
        split=args.val_split,
        context_sec=cfg["data"]["context_sec"],
        patch_sec=cfg["data"]["patch_sec"],
        max_channels=cfg["data"]["max_channels"],
    )

    ds_test = SeizureWindowDataset(
        manifest_csv=cfg["data"]["manifest_csv"],
        split=args.test_split,
        context_sec=cfg["data"]["context_sec"],
        patch_sec=cfg["data"]["patch_sec"],
        max_channels=cfg["data"]["max_channels"],
    )

    print("Train/support pool size:", len(ds_train))
    print("Validation query size:", len(ds_val))
    print("Test query size:", len(ds_test))

    print("\nExtracting embeddings...")
    X_train, y_train = extract_embeddings(model, ds_train, device, batch_size=args.batch_size)
    X_val, y_val = extract_embeddings(model, ds_val, device, batch_size=args.batch_size)
    X_test, y_test = extract_embeddings(model, ds_test, device, batch_size=args.batch_size)

    print("Train embeddings:", X_train.shape, "class counts:", dict(pd.Series(y_train).value_counts()))
    print("Val embeddings:", X_val.shape, "class counts:", dict(pd.Series(y_val).value_counts()))
    print("Test embeddings:", X_test.shape, "class counts:", dict(pd.Series(y_test).value_counts()))

    # Standardize using support pool only
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    rows = []

    tsne_support_X = {}
    tsne_support_y = {}
    first_seed = args.seeds[0]

    for shot in args.shots:
        for seed in args.seeds:
            rng = np.random.default_rng(seed)

            support_idx = make_support_indices(y_train, shot, rng)

            X_support = X_train[support_idx]
            y_support = y_train[support_idx]

            prototypes = compute_prototypes(X_support, y_support)

            score_val, _, _, _ = prototype_scores(X_val, prototypes)
            score_test, _, _, _ = prototype_scores(X_test, prototypes)

            best_threshold = select_threshold_by_validation(y_val, score_val)

            val_metrics = metrics_from_scores(y_val, score_val, best_threshold)
            test_metrics = metrics_from_scores(y_test, score_test, best_threshold)

            val_metrics.update({
                "split": "val",
                "shot": shot,
                "seed": seed,
                "threshold": best_threshold,
            })

            test_metrics.update({
                "split": "test",
                "shot": shot,
                "seed": seed,
                "threshold": best_threshold,
            })

            rows.append(val_metrics)
            rows.append(test_metrics)

            print(f"Shot={shot}, seed={seed}, threshold={best_threshold:.3f}, "
                  f"val_BA={val_metrics['balanced_accuracy']:.3f}, "
                  f"test_BA={test_metrics['balanced_accuracy']:.3f}, "
                  f"test_AUROC={test_metrics['auroc']:.3f}")

            if seed == first_seed:
                tsne_support_X[shot] = X_support
                tsne_support_y[shot] = y_support

    per_seed = pd.DataFrame(rows)
    per_seed_path = out_dir / "prototype_fewshot_per_seed_metrics.csv"
    per_seed.to_csv(per_seed_path, index=False)
    print("Saved:", per_seed_path)

    summary = summarize_mean_std(per_seed)
    summary_path = out_dir / "prototype_fewshot_summary_mean_std.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    # Pretty manuscript-style table
    metric_order = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "sensitivity",
        "specificity",
        "f1_score",
        "f2_score",
        "kappa",
        "mcc",
        "auroc",
        "auprc",
    ]

    pretty_rows = []

    for _, r in summary.iterrows():
        row = {
            "split": r["split"],
            "shot": int(r["shot"]),
            "n_runs": int(r["n_runs"]),
        }

        for m in metric_order:
            row[m] = f"{r[m + '_mean']:.4f} � {r[m + '_std']:.4f}"

        pretty_rows.append(row)

    pretty = pd.DataFrame(pretty_rows)
    pretty_path = out_dir / "prototype_fewshot_summary_pretty.csv"
    pretty.to_csv(pretty_path, index=False)
    print("Saved:", pretty_path)

    # Figures
    plot_metric_small_multiples(summary, out_dir, split="test")
    plot_metric_small_multiples(summary, out_dir, split="val")
    plot_auc_only(summary, out_dir, split="test")
    plot_auc_only(summary, out_dir, split="val")

    # t-SNE for first seed only, both shots
    max_n = min(args.max_query_points_tsne, len(X_test))
    rng = np.random.default_rng(123)
    idx_test = rng.choice(np.arange(len(X_test)), size=max_n, replace=False) if len(X_test) > max_n else np.arange(len(X_test))

    plot_tsne_two_shots(
        X_query=X_test[idx_test],
        y_query=y_test[idx_test],
        X_support_by_shot=tsne_support_X,
        y_support_by_shot=tsne_support_y,
        out_dir=out_dir,
        split="test",
        seed=first_seed,
    )

    max_n_val = min(args.max_query_points_tsne, len(X_val))
    idx_val = rng.choice(np.arange(len(X_val)), size=max_n_val, replace=False) if len(X_val) > max_n_val else np.arange(len(X_val))

    plot_tsne_two_shots(
        X_query=X_val[idx_val],
        y_query=y_val[idx_val],
        X_support_by_shot=tsne_support_X,
        y_support_by_shot=tsne_support_y,
        out_dir=out_dir,
        split="val",
        seed=first_seed,
    )

    print("\nDone. Prototype few-shot results saved in:")
    print(out_dir)


if __name__ == "__main__":
    main()