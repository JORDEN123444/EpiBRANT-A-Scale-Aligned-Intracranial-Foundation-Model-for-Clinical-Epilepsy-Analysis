
from pathlib import Path

import numpy as np

import pandas as pd

import matplotlib.pyplot as plt



from sklearn.metrics import (

    accuracy_score,

    balanced_accuracy_score,

    precision_score,

    recall_score,

    f1_score,

    cohen_kappa_score,

    matthews_corrcoef,

    roc_auc_score,

    average_precision_score,

    brier_score_loss,

    confusion_matrix,

)



RUN_DIR = Path("/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection/runs_patientwise_5s_trainP2P3P4P6_valP5_testP1/pretrained_full_finetune")

FIG_DIR = RUN_DIR / "nature_statistical_figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)



VAL_METRIC_CSV = RUN_DIR / "val_metrics_threshold_optimized.csv"

TEST_METRIC_CSV = RUN_DIR / "test_metrics_threshold_optimized.csv"

VAL_PRED_CSV = RUN_DIR / "val_predictions_best.csv"

TEST_PRED_CSV = RUN_DIR / "test_predictions.csv"



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





def save_all(fig, name):

    base = FIG_DIR / name

    fig.savefig(str(base) + ".png", dpi=300, bbox_inches="tight")

    fig.savefig(str(base) + ".tiff", dpi=600, bbox_inches="tight")

    fig.savefig(str(base) + ".pdf", dpi=600, bbox_inches="tight")

    print("Saved:", str(base) + ".png")

    print("Saved:", str(base) + ".tiff")

    print("Saved:", str(base) + ".pdf")





def find_cols(df):

    y_col = None

    p_col = None



    for c in df.columns:

        cl = c.lower()

        if cl in ["y_true", "label", "true", "target"]:

            y_col = c

        if cl in ["y_prob", "prob", "score", "prediction_prob", "pred_prob"]:

            p_col = c



    if y_col is None:

        for c in df.columns:

            if "true" in c.lower() or "label" in c.lower():

                y_col = c

                break



    if p_col is None:

        for c in df.columns:

            if "prob" in c.lower() or "score" in c.lower():

                p_col = c

                break



    if y_col is None or p_col is None:

        raise RuntimeError("Could not infer label/probability columns: " + str(df.columns.tolist()))



    return y_col, p_col





def metrics_at_threshold(y, p, th):

    pred = (p >= th).astype(int)



    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()



    out = {

        "accuracy": accuracy_score(y, pred),

        "balanced_accuracy": balanced_accuracy_score(y, pred),

        "precision": precision_score(y, pred, zero_division=0),

        "sensitivity": recall_score(y, pred, zero_division=0),

        "specificity": tn / max(1, tn + fp),

        "f1_score": f1_score(y, pred, zero_division=0),

        "kappa": cohen_kappa_score(y, pred),

        "mcc": matthews_corrcoef(y, pred),

        "false_positive_rate": fp / max(1, fp + tn),

        "false_negative_rate": fn / max(1, fn + tp),

        "tn": tn,

        "fp": fp,

        "fn": fn,

        "tp": tp,

    }



    try:

        out["auroc"] = roc_auc_score(y, p)

    except Exception:

        out["auroc"] = np.nan



    try:

        out["auprc"] = average_precision_score(y, p)

    except Exception:

        out["auprc"] = np.nan



    try:

        out["brier"] = brier_score_loss(y, p)

    except Exception:

        out["brier"] = np.nan



    out["youden_j"] = out["sensitivity"] + out["specificity"] - 1.0

    return out





def threshold_table(y, p):

    rows = []

    for th in np.linspace(0.0, 1.0, 101):

        m = metrics_at_threshold(y, p, th)

        m["threshold"] = th

        rows.append(m)

    return pd.DataFrame(rows)





def bootstrap_ci(y, p, th, n_boot=500, seed=42):

    rng = np.random.default_rng(seed)

    n = len(y)



    keys = [

        "accuracy",

        "balanced_accuracy",

        "f1_score",

        "precision",

        "sensitivity",

        "specificity",

        "kappa",

        "mcc",

        "auroc",

        "auprc",

        "brier",

    ]



    boot = {k: [] for k in keys}



    for _ in range(n_boot):

        idx = rng.integers(0, n, size=n)

        yy = y[idx]

        pp = p[idx]



        if len(np.unique(yy)) < 2:

            continue



        m = metrics_at_threshold(yy, pp, th)



        for k in keys:

            boot[k].append(m[k])



    rows = []

    for k in keys:

        vals = np.asarray(boot[k], dtype=float)

        vals = vals[~np.isnan(vals)]



        if len(vals) == 0:

            rows.append({"metric": k, "mean": np.nan, "low": np.nan, "high": np.nan})

        else:

            rows.append({

                "metric": k,

                "mean": np.mean(vals),

                "low": np.percentile(vals, 2.5),

                "high": np.percentile(vals, 97.5),

            })



    return pd.DataFrame(rows)





def calibration_points(y, p, n_bins=10):

    bins = np.linspace(0, 1, n_bins + 1)

    rows = []



    for i in range(n_bins):

        lo, hi = bins[i], bins[i + 1]



        if i == n_bins - 1:

            mask = (p >= lo) & (p <= hi)

        else:

            mask = (p >= lo) & (p < hi)



        if mask.sum() == 0:

            continue



        rows.append({

            "bin_low": lo,

            "bin_high": hi,

            "mean_predicted": float(p[mask].mean()),

            "observed_fraction": float(y[mask].mean()),

            "n": int(mask.sum()),

        })



    return pd.DataFrame(rows)





def ecdf(x):

    x = np.sort(np.asarray(x))

    y = np.arange(1, len(x) + 1) / len(x)

    return x, y





# =========================================================

# Load data

# =========================================================

val_m = pd.read_csv(VAL_METRIC_CSV).iloc[0]

test_m = pd.read_csv(TEST_METRIC_CSV).iloc[0]



threshold = float(val_m["threshold"])



val_pred = pd.read_csv(VAL_PRED_CSV)

test_pred = pd.read_csv(TEST_PRED_CSV)



val_y_col, val_p_col = find_cols(val_pred)

test_y_col, test_p_col = find_cols(test_pred)



y_val = val_pred[val_y_col].values.astype(int)

p_val = val_pred[val_p_col].values.astype(float)



y_test = test_pred[test_y_col].values.astype(int)

p_test = test_pred[test_p_col].values.astype(float)



print("Using validation threshold:", threshold)

print("Validation columns:", val_y_col, val_p_col)

print("Test columns:", test_y_col, test_p_col)



# Save threshold tables

val_thr = threshold_table(y_val, p_val)

test_thr = threshold_table(y_test, p_test)



val_thr.to_csv(FIG_DIR / "validation_threshold_sweep.csv", index=False)

test_thr.to_csv(FIG_DIR / "test_threshold_sweep.csv", index=False)



# =========================================================

# Figure 1: Probability distribution by class

# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4), sharey=True)



for ax, y, p, title in [

    (axes[0], y_val, p_val, "Validation"),

    (axes[1], y_test, p_test, "Test"),

]:

    normal = p[y == 0]

    seizure = p[y == 1]



    vp = ax.violinplot([normal, seizure], positions=[0, 1], showmeans=False, showmedians=True, widths=0.75)

    ax.boxplot([normal, seizure], positions=[0, 1], widths=0.20, showfliers=False)



    ax.axhline(threshold, linestyle="--", linewidth=1.0)

    ax.set_xticks([0, 1])

    ax.set_xticklabels(["Normal", "Seizure"])

    ax.set_ylabel("Predicted seizure probability")

    ax.set_ylim(-0.02, 1.02)

    ax.set_title(title)



fig.tight_layout()

save_all(fig, "fig1_probability_distribution_by_class")

plt.close(fig)



# =========================================================

# Figure 2: Threshold sweep curves

# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.6), sharey=True)



for ax, tbl, title in [

    (axes[0], val_thr, "Validation"),

    (axes[1], test_thr, "Test"),

]:

    ax.plot(tbl["threshold"], tbl["balanced_accuracy"], linewidth=1.5, label="Balanced accuracy")

    ax.plot(tbl["threshold"], tbl["f1_score"], linewidth=1.5, label="F1-score")

    ax.plot(tbl["threshold"], tbl["sensitivity"], linewidth=1.5, label="Sensitivity")

    ax.plot(tbl["threshold"], tbl["specificity"], linewidth=1.5, label="Specificity")

    ax.axvline(threshold, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Decision threshold")

    ax.set_ylabel("Score")

    ax.set_ylim(0, 1.05)

    ax.set_title(title)



axes[1].legend(frameon=False, loc="lower right")

fig.tight_layout()

save_all(fig, "fig2_threshold_sweep_validation_test")

plt.close(fig)



# =========================================================

# Figure 3: Sensitivity-specificity trade-off

# =========================================================

fig, ax = plt.subplots(figsize=(4.8, 4.0))



ax.plot(val_thr["specificity"], val_thr["sensitivity"], linewidth=1.8, label="Validation")

ax.plot(test_thr["specificity"], test_thr["sensitivity"], linewidth=1.8, label="Test")



val_sel = metrics_at_threshold(y_val, p_val, threshold)

test_sel = metrics_at_threshold(y_test, p_test, threshold)



ax.scatter([val_sel["specificity"]], [val_sel["sensitivity"]], s=40, marker="o", label="Selected validation threshold")

ax.scatter([test_sel["specificity"]], [test_sel["sensitivity"]], s=40, marker="^", label="Test at selected threshold")



ax.set_xlabel("Specificity")

ax.set_ylabel("Sensitivity")

ax.set_xlim(0, 1.02)

ax.set_ylim(0, 1.02)

ax.legend(frameon=False)

ax.set_title("Sensitivity-specificity trade-off")



fig.tight_layout()

save_all(fig, "fig3_sensitivity_specificity_tradeoff")

plt.close(fig)



# =========================================================

# Figure 4: Bootstrap confidence intervals

# =========================================================

val_ci = bootstrap_ci(y_val, p_val, threshold, n_boot=500, seed=42)

test_ci = bootstrap_ci(y_test, p_test, threshold, n_boot=500, seed=43)



val_ci["split"] = "Validation"

test_ci["split"] = "Test"



ci_all = pd.concat([val_ci, test_ci], ignore_index=True)

ci_all.to_csv(FIG_DIR / "bootstrap_95ci_metrics.csv", index=False)



plot_metrics = [

    "balanced_accuracy",

    "f1_score",

    "precision",

    "sensitivity",

    "specificity",

    "auroc",

    "auprc",

]



labels = [

    "Balanced acc.",

    "F1-score",

    "Precision",

    "Sensitivity",

    "Specificity",

    "AUROC",

    "AUPRC",

]



fig, ax = plt.subplots(figsize=(6.6, 4.2))



ypos = np.arange(len(plot_metrics))



for offset, split in [(-0.12, "Validation"), (0.12, "Test")]:

    d = ci_all[(ci_all["split"] == split) & (ci_all["metric"].isin(plot_metrics))].copy()

    d["order"] = d["metric"].apply(lambda x: plot_metrics.index(x))

    d = d.sort_values("order")



    means = d["mean"].values

    lows = d["low"].values

    highs = d["high"].values



    xerr = np.vstack([means - lows, highs - means])



    ax.errorbar(

        means,

        ypos + offset,

        xerr=xerr,

        fmt="o",

        capsize=3,

        linewidth=1.2,

        label=split,

    )



ax.set_yticks(ypos)

ax.set_yticklabels(labels)

ax.set_xlabel("Metric value with bootstrap 95% CI")

ax.set_xlim(0, 1.05)

ax.invert_yaxis()

ax.legend(frameon=False)

ax.set_title("Bootstrap uncertainty of performance metrics")



fig.tight_layout()

save_all(fig, "fig4_bootstrap_confidence_intervals")

plt.close(fig)



# =========================================================

# Figure 5: Calibration reliability plot

# =========================================================

cal_val = calibration_points(y_val, p_val, n_bins=10)

cal_test = calibration_points(y_test, p_test, n_bins=10)



cal_val.to_csv(FIG_DIR / "validation_calibration_bins.csv", index=False)

cal_test.to_csv(FIG_DIR / "test_calibration_bins.csv", index=False)



brier_val = brier_score_loss(y_val, p_val)

brier_test = brier_score_loss(y_test, p_test)



fig, ax = plt.subplots(figsize=(4.8, 4.0))



ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0, label="Perfect calibration")



if len(cal_val) > 0:

    ax.plot(cal_val["mean_predicted"], cal_val["observed_fraction"], marker="o", linewidth=1.5, label=f"Validation, Brier={brier_val:.3f}")



if len(cal_test) > 0:

    ax.plot(cal_test["mean_predicted"], cal_test["observed_fraction"], marker="^", linewidth=1.5, label=f"Test, Brier={brier_test:.3f}")



ax.set_xlabel("Mean predicted seizure probability")

ax.set_ylabel("Observed seizure fraction")

ax.set_xlim(0, 1)

ax.set_ylim(0, 1)

ax.legend(frameon=False)

ax.set_title("Calibration reliability")



fig.tight_layout()

save_all(fig, "fig5_calibration_reliability")

plt.close(fig)



# =========================================================

# Figure 6: Normalized confusion matrices

# =========================================================

cm_val = confusion_matrix(y_val, (p_val >= threshold).astype(int), labels=[0, 1])

cm_test = confusion_matrix(y_test, (p_test >= threshold).astype(int), labels=[0, 1])



cm_val_norm = cm_val / cm_val.sum(axis=1, keepdims=True)

cm_test_norm = cm_test / cm_test.sum(axis=1, keepdims=True)



fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))



for ax, cm, title in [

    (axes[0], cm_val_norm, "Validation"),

    (axes[1], cm_test_norm, "Test"),

]:

    ax.imshow(cm, vmin=0, vmax=1)

    ax.set_xticks([0, 1])

    ax.set_yticks([0, 1])

    ax.set_xticklabels(["Normal", "Seizure"])

    ax.set_yticklabels(["Normal", "Seizure"])

    ax.set_xlabel("Predicted label")

    ax.set_ylabel("True label")

    ax.set_title(title)



    for i in range(2):

        for j in range(2):

            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=10)



fig.tight_layout()

save_all(fig, "fig6_normalized_confusion_matrices")

plt.close(fig)



# =========================================================

# Figure 7: Empirical cumulative distribution of probabilities

# =========================================================

fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.4), sharey=True)



for ax, y, p, title in [

    (axes[0], y_val, p_val, "Validation"),

    (axes[1], y_test, p_test, "Test"),

]:

    for label, name in [(0, "Normal"), (1, "Seizure")]:

        x_ecdf, y_ecdf = ecdf(p[y == label])

        ax.plot(x_ecdf, y_ecdf, linewidth=1.5, label=name)



    ax.axvline(threshold, linestyle="--", linewidth=1.0)

    ax.set_xlabel("Predicted seizure probability")

    ax.set_ylabel("Cumulative fraction")

    ax.set_xlim(0, 1)

    ax.set_ylim(0, 1.02)

    ax.set_title(title)



axes[1].legend(frameon=False)

fig.tight_layout()

save_all(fig, "fig7_ecdf_prediction_scores")

plt.close(fig)



# =========================================================

# Figure 8: Error composition

# =========================================================

error_df = pd.DataFrame([

    {"split": "Validation", "type": "True normal", "count": int(val_sel["tn"])},

    {"split": "Validation", "type": "False alarm", "count": int(val_sel["fp"])},

    {"split": "Validation", "type": "Missed seizure", "count": int(val_sel["fn"])},

    {"split": "Validation", "type": "Detected seizure", "count": int(val_sel["tp"])},

    {"split": "Test", "type": "True normal", "count": int(test_sel["tn"])},

    {"split": "Test", "type": "False alarm", "count": int(test_sel["fp"])},

    {"split": "Test", "type": "Missed seizure", "count": int(test_sel["fn"])},

    {"split": "Test", "type": "Detected seizure", "count": int(test_sel["tp"])},

])



error_df.to_csv(FIG_DIR / "error_composition_counts.csv", index=False)



types = ["True normal", "False alarm", "Missed seizure", "Detected seizure"]

x = np.arange(len(types))

width = 0.36



val_counts = [error_df[(error_df["split"] == "Validation") & (error_df["type"] == t)]["count"].iloc[0] for t in types]

test_counts = [error_df[(error_df["split"] == "Test") & (error_df["type"] == t)]["count"].iloc[0] for t in types]



fig, ax = plt.subplots(figsize=(6.8, 3.8))

ax.bar(x - width/2, val_counts, width, label="Validation")

ax.bar(x + width/2, test_counts, width, label="Test")

ax.set_xticks(x)

ax.set_xticklabels(types, rotation=25, ha="right")

ax.set_ylabel("Number of windows")

ax.legend(frameon=False)

ax.set_title("Error composition at validation-selected threshold")



fig.tight_layout()

save_all(fig, "fig8_error_composition")

plt.close(fig)



print("\nDone. Extra statistical figures saved in:")

print(FIG_DIR)

