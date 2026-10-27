
from pathlib import Path



import numpy as np

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, precision_recall_curve, auc, average_precision_score



# =========================================================

# Paths

# =========================================================

RUN_DIR = Path("/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection/runs_patientwise_5s_trainP2P3P4P6_valP5_testP1/pretrained_full_finetune")

FIG_DIR = RUN_DIR / "final_figures_validation_based"

FIG_DIR.mkdir(parents=True, exist_ok=True)



VAL_METRIC_CSV = RUN_DIR / "val_metrics_threshold_optimized.csv"

TEST_METRIC_CSV = RUN_DIR / "test_metrics_threshold_optimized.csv"



VAL_PRED_CSV = RUN_DIR / "val_predictions_best.csv"

TEST_PRED_CSV = RUN_DIR / "test_predictions.csv"



TRAIN_LOG_CSV = RUN_DIR / "training_log.csv"



# =========================================================

# Plot style

# =========================================================

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



def save_all(fig, out_base):

    fig.savefig(str(out_base) + ".png", dpi=300, bbox_inches="tight")

    fig.savefig(str(out_base) + ".tiff", dpi=600, bbox_inches="tight")

    fig.savefig(str(out_base) + ".pdf", dpi=600, bbox_inches="tight")

    print("Saved:", str(out_base) + ".png")

    print("Saved:", str(out_base) + ".tiff")

    print("Saved:", str(out_base) + ".pdf")



def find_cols(df):

    y_true_col = None

    y_prob_col = None



    for c in df.columns:

        cl = c.lower()

        if cl in ["y_true", "label", "true", "target"]:

            y_true_col = c

        if cl in ["y_prob", "prob", "score", "prediction_prob", "pred_prob"]:

            y_prob_col = c



    if y_true_col is None:

        for c in df.columns:

            if "true" in c.lower() or "label" in c.lower():

                y_true_col = c

                break



    if y_prob_col is None:

        for c in df.columns:

            if "prob" in c.lower() or "score" in c.lower():

                y_prob_col = c

                break



    if y_true_col is None or y_prob_col is None:

        raise RuntimeError(

            f"Could not infer columns from {df.columns.tolist()}. "

            f"Need one true-label column and one probability column."

        )



    return y_true_col, y_prob_col



# =========================================================

# Load data

# =========================================================

val_m = pd.read_csv(VAL_METRIC_CSV).iloc[0]

test_m = pd.read_csv(TEST_METRIC_CSV).iloc[0]



val_pred = pd.read_csv(VAL_PRED_CSV)

test_pred = pd.read_csv(TEST_PRED_CSV)



val_true_col, val_prob_col = find_cols(val_pred)

test_true_col, test_prob_col = find_cols(test_pred)



y_val = val_pred[val_true_col].values

p_val = val_pred[val_prob_col].values



y_test = test_pred[test_true_col].values

p_test = test_pred[test_prob_col].values



threshold = float(val_m["threshold"])



print("Validation prediction columns:", val_pred.columns.tolist())

print("Test prediction columns:", test_pred.columns.tolist())

print("Using threshold:", threshold)



# =========================================================

# Figure 1 — ROC curve

# =========================================================

fig, ax = plt.subplots(figsize=(6.4, 4.2))



fpr_val, tpr_val, _ = roc_curve(y_val, p_val)

fpr_test, tpr_test, _ = roc_curve(y_test, p_test)



auc_val = auc(fpr_val, tpr_val)

auc_test = auc(fpr_test, tpr_test)



ax.plot(fpr_val, tpr_val, linewidth=1.8, label=f"Validation (AUROC={auc_val:.3f})")

ax.plot(fpr_test, tpr_test, linewidth=1.8, label=f"Test (AUROC={auc_test:.3f})")

ax.plot([0, 1], [0, 1], linestyle="--", linewidth=0.8)



ax.set_xlabel("False positive rate")

ax.set_ylabel("True positive rate")

ax.set_xlim(0, 1)

ax.set_ylim(0, 1.02)

ax.legend(frameon=False)

ax.set_title("ROC curves")



fig.tight_layout()

save_all(fig, FIG_DIR / "fig1_roc_validation_test")

plt.close(fig)



# =========================================================

# Figure 2 — Precision-Recall curve

# =========================================================

fig, ax = plt.subplots(figsize=(6.4, 4.2))



prec_val, rec_val, _ = precision_recall_curve(y_val, p_val)

prec_test, rec_test, _ = precision_recall_curve(y_test, p_test)



ap_val = average_precision_score(y_val, p_val)

ap_test = average_precision_score(y_test, p_test)



ax.plot(rec_val, prec_val, linewidth=1.8, label=f"Validation (AUPRC={ap_val:.3f})")

ax.plot(rec_test, prec_test, linewidth=1.8, label=f"Test (AUPRC={ap_test:.3f})")



ax.set_xlabel("Recall")

ax.set_ylabel("Precision")

ax.set_xlim(0, 1)

ax.set_ylim(0, 1.02)

ax.legend(frameon=False)

ax.set_title("Precision–Recall curves")



fig.tight_layout()

save_all(fig, FIG_DIR / "fig2_pr_validation_test")

plt.close(fig)



# =========================================================

# Figure 3 — Validation confusion matrix

# =========================================================

cm_val = np.array([

    [int(val_m["tn"]), int(val_m["fp"])],

    [int(val_m["fn"]), int(val_m["tp"])]

])



fig, ax = plt.subplots(figsize=(4.0, 3.6))

ax.imshow(cm_val, interpolation="nearest")

ax.set_xticks([0, 1])

ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Seizure"])

ax.set_yticklabels(["Normal", "Seizure"])

ax.set_xlabel("Predicted label")

ax.set_ylabel("True label")

ax.set_title(f"Validation confusion matrix\nThreshold = {threshold:.2f}")



for i in range(2):

    for j in range(2):

        ax.text(j, i, str(cm_val[i, j]), ha="center", va="center", fontsize=11)



fig.tight_layout()

save_all(fig, FIG_DIR / "fig3_confusion_validation_threshold_0p03")

plt.close(fig)



# =========================================================

# Figure 4 — Test confusion matrix

# =========================================================

cm_test = np.array([

    [int(test_m["tn"]), int(test_m["fp"])],

    [int(test_m["fn"]), int(test_m["tp"])]

])



fig, ax = plt.subplots(figsize=(4.0, 3.6))

ax.imshow(cm_test, interpolation="nearest")

ax.set_xticks([0, 1])

ax.set_yticks([0, 1])

ax.set_xticklabels(["Normal", "Seizure"])

ax.set_yticklabels(["Normal", "Seizure"])

ax.set_xlabel("Predicted label")

ax.set_ylabel("True label")

ax.set_title(f"Test confusion matrix\nValidation-selected threshold = {threshold:.2f}")



for i in range(2):

    for j in range(2):

        ax.text(j, i, str(cm_test[i, j]), ha="center", va="center", fontsize=11)



fig.tight_layout()

save_all(fig, FIG_DIR / "fig4_confusion_test_threshold_0p03")

plt.close(fig)



# =========================================================

# Figure 5 — Validation vs test metric bar chart

# =========================================================

metrics = [

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



metric_labels = [

    "Accuracy",

    "Balanced acc.",

    "F1-score",

    "Precision",

    "Sensitivity",

    "Specificity",

    "Kappa",

    "AUROC",

    "AUPRC",

]



val_vals = [float(val_m[m]) for m in metrics]

test_vals = [float(test_m[m]) for m in metrics]



x = np.arange(len(metrics))

width = 0.36



fig, ax = plt.subplots(figsize=(8.8, 4.2))

ax.bar(x - width/2, val_vals, width, label="Validation")

ax.bar(x + width/2, test_vals, width, label="Test")



ax.set_xticks(x)

ax.set_xticklabels(metric_labels, rotation=30, ha="right")

ax.set_ylabel("Score")

ax.set_ylim(0, 1.05)

ax.legend(frameon=False)

ax.set_title("Validation-selected threshold performance")



fig.tight_layout()

save_all(fig, FIG_DIR / "fig5_metric_bar_validation_vs_test")

plt.close(fig)



# =========================================================

# Figure 6 — AUROC and AUPRC only

# =========================================================

auc_metrics = ["auroc", "auprc"]

auc_labels = ["AUROC", "AUPRC"]



val_auc = [float(val_m[m]) for m in auc_metrics]

test_auc = [float(test_m[m]) for m in auc_metrics]



x = np.arange(len(auc_metrics))

width = 0.36



fig, ax = plt.subplots(figsize=(4.8, 3.6))

ax.bar(x - width/2, val_auc, width, label="Validation")

ax.bar(x + width/2, test_auc, width, label="Test")



ax.set_xticks(x)

ax.set_xticklabels(auc_labels)

ax.set_ylabel("Score")

ax.set_ylim(0, 1.05)

ax.legend(frameon=False)

ax.set_title("AUROC and AUPRC comparison")



for i, v in enumerate(val_auc):

    ax.text(i - width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)



for i, v in enumerate(test_auc):

    ax.text(i + width/2, v + 0.02, f"{v:.3f}", ha="center", fontsize=7)



fig.tight_layout()

save_all(fig, FIG_DIR / "fig6_auroc_auprc_comparison")

plt.close(fig)



# =========================================================

# Figure 7 — Training dynamics

# =========================================================

if TRAIN_LOG_CSV.exists():

    log = pd.read_csv(TRAIN_LOG_CSV)



    fig, ax = plt.subplots(figsize=(6.8, 4.2))



    if "train_loss" in log.columns:

        ax.plot(log["epoch"], log["train_loss"], linewidth=1.5, label="Train loss")

    if "val_balanced_accuracy" in log.columns:

        ax.plot(log["epoch"], log["val_balanced_accuracy"], linewidth=1.5, label="Val balanced acc.")

    if "val_auroc" in log.columns:

        ax.plot(log["epoch"], log["val_auroc"], linewidth=1.5, label="Val AUROC")

    if "val_auprc" in log.columns:

        ax.plot(log["epoch"], log["val_auprc"], linewidth=1.5, label="Val AUPRC")



    ax.set_xlabel("Epoch")

    ax.set_ylabel("Value")

    ax.legend(frameon=False)

    ax.set_title("Fine-tuning dynamics")



    fig.tight_layout()

    save_all(fig, FIG_DIR / "fig7_training_dynamics")

    plt.close(fig)



# =========================================================

# Save summary table

# =========================================================

summary = pd.DataFrame([

    ["Validation", threshold, val_m["accuracy"], val_m["balanced_accuracy"], val_m["f1_score"], val_m["precision"], val_m["sensitivity"], val_m["specificity"], val_m["kappa"], val_m["auroc"], val_m["auprc"], int(val_m["tn"]), int(val_m["fp"]), int(val_m["fn"]), int(val_m["tp"])],

    ["Test", threshold, test_m["accuracy"], test_m["balanced_accuracy"], test_m["f1_score"], test_m["precision"], test_m["sensitivity"], test_m["specificity"], test_m["kappa"], test_m["auroc"], test_m["auprc"], int(test_m["tn"]), int(test_m["fp"]), int(test_m["fn"]), int(test_m["tp"])],

], columns=[

    "split", "threshold", "accuracy", "balanced_accuracy", "f1_score", "precision",

    "sensitivity", "specificity", "kappa", "auroc", "auprc", "tn", "fp", "fn", "tp"

])



summary.to_csv(FIG_DIR / "summary_validation_based_metrics.csv", index=False)

print("Saved:", FIG_DIR / "summary_validation_based_metrics.csv")



print("\nDone. All figures saved in:")

print(FIG_DIR)

