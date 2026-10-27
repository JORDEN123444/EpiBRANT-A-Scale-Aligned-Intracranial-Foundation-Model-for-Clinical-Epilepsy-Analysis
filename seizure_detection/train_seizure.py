
import argparse

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))



import numpy as np

import pandas as pd

import torch

import torch.nn.functional as F

import yaml

from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    f1_score,

    precision_score,

    recall_score,

    confusion_matrix,

    roc_auc_score,

    average_precision_score,

    cohen_kappa_score,

)

from torch.utils.data import DataLoader, WeightedRandomSampler



from downstream.seizure_detection.data_seizure import SeizureWindowDataset

from downstream.seizure_detection.model_seizure import SABrantSeizureClassifier





def compute_metrics(y_true, y_prob, threshold=0.5):

    y_true = np.asarray(y_true).astype(int)

    y_prob = np.asarray(y_prob).astype(float)

    y_pred = (y_prob >= threshold).astype(int)



    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    tn, fp, fn, tp = cm.ravel()



    sensitivity = tp / max(1, tp + fn)

    specificity = tn / max(1, tn + fp)



    out = {

        "accuracy": accuracy_score(y_true, y_pred),

        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),

        "f1_score": f1_score(y_true, y_pred, zero_division=0),

        "precision": precision_score(y_true, y_pred, zero_division=0),

        "sensitivity": sensitivity,

        "specificity": specificity,

        "kappa": cohen_kappa_score(y_true, y_pred),

        "tn": int(tn),

        "fp": int(fp),

        "fn": int(fn),

        "tp": int(tp),

    }



    if len(np.unique(y_true)) == 2:

        out["auroc"] = roc_auc_score(y_true, y_prob)

        out["auprc"] = average_precision_score(y_true, y_prob)

    else:

        out["auroc"] = np.nan

        out["auprc"] = np.nan



    return out





def evaluate(model, loader, device, amp_enabled):

    model.eval()

    losses = []

    y_true = []

    y_prob = []

    rows = []



    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(device)

            fs = batch["fs"].to(device)

            channel_mask = batch["channel_mask"].to(device)

            label = batch["label"].to(device)



            with torch.cuda.amp.autocast(enabled=amp_enabled):

                logit = model(x, fs, channel_mask)

                loss = F.binary_cross_entropy_with_logits(logit, label)



            prob = torch.sigmoid(logit)



            losses.append(float(loss.detach().cpu()))



            for i in range(len(label)):

                y_true.append(float(label[i].detach().cpu()))

                y_prob.append(float(prob[i].detach().cpu()))

                rows.append({

                    "y_true": float(label[i].detach().cpu()),

                    "y_prob": float(prob[i].detach().cpu()),

                    "run": batch["run"][i],

                    "original_name": batch["original_name"][i],

                    "start_sec": float(batch["start_sec"][i].detach().cpu()),

                })



    metrics = compute_metrics(y_true, y_prob)

    metrics["loss"] = float(np.mean(losses))



    return metrics, pd.DataFrame(rows)





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument("--pretrained", default=None)

    parser.add_argument("--tag", default="run")

    parser.add_argument("--freeze_backbone", action="store_true")

    args = parser.parse_args()



    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)



    out_dir = Path(cfg["train"]["out_dir"]) / args.tag

    out_dir.mkdir(parents=True, exist_ok=True)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"



    train_ds = SeizureWindowDataset(

        cfg["data"]["manifest_csv"],

        split="train",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        seed=cfg["train"]["seed"],

    )



    val_ds = SeizureWindowDataset(

        cfg["data"]["manifest_csv"],

        split="val",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        seed=cfg["train"]["seed"] + 1000,

    )



    test_ds = SeizureWindowDataset(

        cfg["data"]["manifest_csv"],

        split="test",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        seed=cfg["train"]["seed"] + 2000,

    )



    labels = train_ds.df["label"].values.astype(int)

    class_counts = np.bincount(labels, minlength=2)



    class_weights = 1.0 / np.maximum(class_counts, 1)

    sample_weights = class_weights[labels]



    sampler = WeightedRandomSampler(

        weights=torch.DoubleTensor(sample_weights),

        num_samples=len(sample_weights),

        replacement=True,

    )



    train_loader = DataLoader(

        train_ds,

        batch_size=cfg["train"]["batch_size"],

        sampler=sampler,

        num_workers=cfg["train"]["num_workers"],

    )



    val_loader = DataLoader(

        val_ds,

        batch_size=cfg["train"]["batch_size"],

        shuffle=False,

        num_workers=cfg["train"]["num_workers"],

    )



    test_loader = DataLoader(

        test_ds,

        batch_size=cfg["train"]["batch_size"],

        shuffle=False,

        num_workers=cfg["train"]["num_workers"],

    )



    model = SABrantSeizureClassifier(

        pretrained_path=args.pretrained,

        freeze_backbone=args.freeze_backbone,

        d_model=cfg["model"]["d_model"],

        seq_len=cfg["data"]["context_sec"] // cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        temporal_layers=cfg["model"]["temporal_layers"],

        channel_layers=cfg["model"]["channel_layers"],

        n_heads=cfg["model"]["n_heads"],

        dim_feedforward=cfg["model"]["dim_feedforward"],

        dropout=cfg["model"]["dropout"],

        t_align=cfg["model"]["t_align"],

        f_align=cfg["model"]["f_align"],

        f_max=cfg["model"]["f_max"],

        conv_channels=cfg["model"]["conv_channels"],

    )



    model = model.to(device)



    pos = max(1, class_counts[1])

    neg = max(1, class_counts[0])

    pos_weight = torch.tensor([neg / pos], dtype=torch.float32).to(device)



    if args.freeze_backbone:

        optimizer = torch.optim.AdamW(

            filter(lambda p: p.requires_grad, model.parameters()),

            lr=cfg["train"]["head_lr"],

            weight_decay=cfg["train"]["weight_decay"],

        )

    else:

        optimizer = torch.optim.AdamW([

            {"params": model.backbone.scale_embed.parameters(), "lr": cfg["train"]["backbone_lr"] * 0.5},

            {"params": model.backbone.temporal_encoder.parameters(), "lr": cfg["train"]["backbone_lr"]},

            {"params": model.backbone.channel_encoder.parameters(), "lr": cfg["train"]["backbone_lr"]},

            {"params": model.classifier.parameters(), "lr": cfg["train"]["head_lr"]},

        ], weight_decay=cfg["train"]["weight_decay"])



    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)



    logs = []

    best_score = -1.0



    print("Device:", device)

    print("AMP:", amp_enabled)

    print("Train class counts [normal, seizure]:", class_counts)

    print("Output:", out_dir)



    for epoch in range(1, cfg["train"]["epochs"] + 1):

        model.train()

        total_loss = 0.0

        n = 0



        for batch in train_loader:

            x = batch["x"].to(device)

            fs = batch["fs"].to(device)

            channel_mask = batch["channel_mask"].to(device)

            label = batch["label"].to(device)



            optimizer.zero_grad(set_to_none=True)



            with torch.cuda.amp.autocast(enabled=amp_enabled):

                logit = model(x, fs, channel_mask)

                loss = F.binary_cross_entropy_with_logits(

                    logit,

                    label,

                    pos_weight=pos_weight,

                )



            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])

            scaler.step(optimizer)

            scaler.update()



            total_loss += float(loss.detach().cpu())

            n += 1



        train_loss = total_loss / max(1, n)

        val_metrics, val_pred = evaluate(model, val_loader, device, amp_enabled)



        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            **{f"val_{k}": v for k, v in val_metrics.items()},

        }



        logs.append(row)

        pd.DataFrame(logs).to_csv(out_dir / "training_log.csv", index=False)



        print(row)



        score = val_metrics["balanced_accuracy"]

        if score > best_score:

            best_score = score

            torch.save(model.state_dict(), out_dir / "best_model.pt")

            val_pred.to_csv(out_dir / "val_predictions_best.csv", index=False)



        torch.save(model.state_dict(), out_dir / "last_model.pt")



    best_state = torch.load(out_dir / "best_model.pt", map_location=device)

    model.load_state_dict(best_state, strict=False)



    test_metrics, test_pred = evaluate(model, test_loader, device, amp_enabled)

    test_pred.to_csv(out_dir / "test_predictions.csv", index=False)

    pd.DataFrame([test_metrics]).to_csv(out_dir / "test_metrics.csv", index=False)



    print("\nFinal test metrics:")

    print(test_metrics)

    print("Saved:", out_dir)





if __name__ == "__main__":

    main()

