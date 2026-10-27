
from pathlib import Path

import re

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score





RUN_ROOT = Path("/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection/runs_lopo_patientwise_5s_pretrained")

FIG_DIR = RUN_ROOT / "nature_style_lopo_figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)



plt.rcParams.update({

    "font.family": "sans-serif",

    "font.size": 7,

    "axes.labelsize": 7,

    "xtick.labelsize": 6,

    "ytick.labelsize": 6,

    "legend.fontsize": 6,

    "axes.linewidth": 0.7,

    "pdf.fonttype": 42,

    "ps.fonttype": 42,

})





def save_all(name):

    plt.savefig(FIG_DIR / f"{name}.png", dpi=300, bbox_inches="tight")

    plt.savefig(FIG_DIR / f"{name}.tiff", dpi=600, bbox_inches="tight")

    plt.savefig(FIG_DIR / f"{name}.pdf", dpi=600, bbox_inches="tight")

    print("Saved:", name)





def patient_from_fold_name(name):

    m = re.search(r"fold_test_(Patient_\d+)_val_(Patient_\d+)", name)

    if m:

        return m.group(1), m.group(2)

    return name, "unknown"





fold_dirs = sorted([p for p in RUN_ROOT.iterdir() if p.is_dir() and p.name.startswith("fold_test_")])



records = []

all_preds = []



for fold_dir in fold_dirs:

    run_dir = fold_dir / "pretrained_full_finetune"

    test_patient, val_patient = patient_from_fold_name(fold_dir.name)



    metric_path = run_dir / "test_metrics_threshold_optimized.csv"

    if not metric_path.exists():

        metric_path = run_dir / "test_metrics.csv"



    pred_path = run_dir / "test_predictions.csv"



    if not metric_path.exists():

        print("[Missing metrics]", fold_dir)

        continue



    m = pd.read_csv(metric_path).iloc[0].to_dict()

    m["fold"] = fold_dir.name

    m["test_patient"] = test_patient

    m["val_patient"] = val_patient

    records.append(m)



    if pred_path.exists():

        pred = pd.read_csv(pred_path)

        pred["fold"] = fold_dir.name

        pred["test_patient"] = test_patient

        all_preds.append(pred)



summary = pd.DataFrame(records)



if len(summary) == 0:

    raise RuntimeError("No fold metrics found. Please finish LOPO training first.")



summary_path = FIG_DIR / "lopo_patientwise_5s_metrics_by_fold.csv"

summary.to_csv(summary_path, index=False)



metrics_for_report = [

    "accuracy",

    "balanced_accuracy",

    "f1_score",

    "precision",

    "sensitivity",

    "specificity",

    "kappa",

    "auroc",

    "auprc",

]



mean_std = summary[metrics_for_report].agg(["mean", "std"]).T

mean_std.to_csv(FIG_DIR / "lopo_patientwise_5s_mean_std_metrics.csv")



print("\nPer-fold summary:")

print(summary[["test_patient"] + metrics_for_report])



print("\nMean ± SD:")

print(mean_std)





# =========================================================

# Figure 1: LOPO metric mean ± SD

# =========================================================

metric_labels = [

    ("balanced_accuracy", "Balanced\naccuracy"),

    ("f1_score", "F1-score"),

    ("auroc", "AUROC"),

    ("auprc", "AUPRC"),

    ("sensitivity", "Sensitivity"),

    ("specificity", "Specificity"),

]



means = [summary[m].mean() for m, _ in metric_labels]

stds = [summary[m].std() for m, _ in metric_labels]



plt.figure(figsize=(6.4, 3.6))

x = np.arange(len(metric_labels))

plt.bar(x, means, yerr=stds, capsize=3)

plt.xticks(x, [lab for _, lab in metric_labels])

plt.ylabel("Score")

plt.ylim(0, 1.05)

plt.tight_layout()

save_all("fig1_lopo_mean_sd_metric_summary")

plt.close()





# =========================================================

# Figure 2: Per-patient balanced accuracy

# =========================================================

order = summary.sort_values("test_patient")

plt.figure(figsize=(6.4, 3.6))

x = np.arange(len(order))

plt.bar(x, order["balanced_accuracy"].values)

plt.xticks(x, order["test_patient"].values, rotation=30, ha="right")

plt.ylabel("Balanced accuracy")

plt.ylim(0, 1.05)

plt.tight_layout()

save_all("fig2_lopo_per_patient_balanced_accuracy")

plt.close()





# =========================================================

# Figure 3: ROC curves for all folds

# =========================================================

plt.figure(figsize=(6.4, 3.6))



if len(all_preds) > 0:

    for pred in all_preds:

        y = pred["y_true"].values

        p = pred["y_prob"].values

        test_patient = pred["test_patient"].iloc[0]



        fpr, tpr, _ = roc_curve(y, p)

        roc_auc = auc(fpr, tpr)

        plt.plot(fpr, tpr, linewidth=1.1, label=f"{test_patient} AUC={roc_auc:.3f}")



    pooled = pd.concat(all_preds, ignore_index=True)

    fpr, tpr, _ = roc_curve(pooled["y_true"].values, pooled["y_prob"].values)

    pooled_auc = auc(fpr, tpr)

    plt.plot(fpr, tpr, linewidth=2.2, label=f"Pooled AUC={pooled_auc:.3f}")



plt.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)

plt.xlabel("False positive rate")

plt.ylabel("True positive rate")

plt.xlim(0, 1)

plt.ylim(0, 1.02)

plt.legend(frameon=False, ncol=2)

plt.tight_layout()

save_all("fig3_lopo_roc_curves")

plt.close()





# =========================================================

# Figure 4: Precision-recall curves for all folds

# =========================================================

plt.figure(figsize=(6.4, 3.6))



if len(all_preds) > 0:

    for pred in all_preds:

        y = pred["y_true"].values

        p = pred["y_prob"].values

        test_patient = pred["test_patient"].iloc[0]



        precision, recall, _ = precision_recall_curve(y, p)

        ap = average_precision_score(y, p)

        plt.plot(recall, precision, linewidth=1.1, label=f"{test_patient} AP={ap:.3f}")



    pooled = pd.concat(all_preds, ignore_index=True)

    precision, recall, _ = precision_recall_curve(pooled["y_true"].values, pooled["y_prob"].values)

    pooled_ap = average_precision_score(pooled["y_true"].values, pooled["y_prob"].values)

    plt.plot(recall, precision, linewidth=2.2, label=f"Pooled AP={pooled_ap:.3f}")



plt.xlabel("Recall")

plt.ylabel("Precision")

plt.xlim(0, 1)

plt.ylim(0, 1.02)

plt.legend(frameon=False, ncol=2)

plt.tight_layout()

save_all("fig4_lopo_precision_recall_curves")

plt.close()





# =========================================================

# Figure 5: Pooled confusion matrix

# =========================================================

tn = int(summary["tn"].sum())

fp = int(summary["fp"].sum())

fn = int(summary["fn"].sum())

tp = int(summary["tp"].sum())



cm = np.array([[tn, fp], [fn, tp]])



plt.figure(figsize=(3.8, 3.5))

plt.imshow(cm, interpolation="nearest")

plt.xticks([0, 1], ["Normal", "Seizure"])

plt.yticks([0, 1], ["Normal", "Seizure"])

plt.xlabel("Predicted label")

plt.ylabel("True label")



for i in range(2):

    for j in range(2):

        plt.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=10)



plt.tight_layout()

save_all("fig5_lopo_pooled_confusion_matrix")

plt.close()





# =========================================================

# Figure 6: Metric heatmap by test patient

# =========================================================

heat_metrics = ["balanced_accuracy", "f1_score", "sensitivity", "specificity", "auroc", "auprc"]

heat = summary.sort_values("test_patient").set_index("test_patient")[heat_metrics]



plt.figure(figsize=(6.4, 3.6))

plt.imshow(heat.values, aspect="auto", vmin=0, vmax=1)

plt.xticks(np.arange(len(heat_metrics)), ["Bal Acc", "F1", "Sens", "Spec", "AUROC", "AUPRC"], rotation=30, ha="right")

plt.yticks(np.arange(len(heat.index)), heat.index)



for i in range(heat.shape[0]):

    for j in range(heat.shape[1]):

        plt.text(j, i, f"{heat.values[i, j]:.2f}", ha="center", va="center", fontsize=6)



plt.colorbar(label="Score")

plt.tight_layout()

save_all("fig6_lopo_patient_metric_heatmap")

plt.close()





# =========================================================

# Figure 7: Training dynamics across folds

# =========================================================

plt.figure(figsize=(6.4, 3.6))

has_log = False



for fold_dir in fold_dirs:

    run_dir = fold_dir / "pretrained_full_finetune"

    log_path = run_dir / "training_log.csv"

    test_patient, _ = patient_from_fold_name(fold_dir.name)



    if not log_path.exists():

        continue



    log = pd.read_csv(log_path)



    if "val_balanced_accuracy" in log.columns:

        plt.plot(log["epoch"], log["val_balanced_accuracy"], linewidth=1.0, label=test_patient)

        has_log = True



if has_log:

    plt.xlabel("Epoch")

    plt.ylabel("Validation balanced accuracy")

    plt.ylim(0, 1.05)

    plt.legend(frameon=False, ncol=2)

    plt.tight_layout()

    save_all("fig7_lopo_validation_dynamics")

plt.close()





print("\nAll LOPO Nature-style figures saved in:")

print(FIG_DIR)

