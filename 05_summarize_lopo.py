
from pathlib import Path

import pandas as pd

import numpy as np

from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, average_precision_score, confusion_matrix, matthews_corrcoef, cohen_kappa_score



SLEEP_ROOT = Path("/home/ubuntu/IJAZ/SEEG datast/finetunedata/Sllepawak daata/New-sleep awak data")

LOPO_DIR = SLEEP_ROOT / "sleep_awake_finetune_outputs" / "lopo_cross_validation"

RUNS_DIR = LOPO_DIR / "runs"



rows = []



for run_dir in sorted(RUNS_DIR.glob("lopo_val_*")):

    pred_path = run_dir / "val_predictions.csv"

    log_path = run_dir / "training_log.csv"



    if not pred_path.exists():

        print("Missing predictions, skip:", run_dir)

        continue



    df = pd.read_csv(pred_path)

    y = df["true_label"].astype(int).values

    p = df["prob_sleep"].astype(float).values

    pred = (p >= 0.5).astype(int)



    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()



    try:

        auroc = roc_auc_score(y, p)

    except Exception:

        auroc = np.nan



    try:

        auprc = average_precision_score(y, p)

    except Exception:

        auprc = np.nan



    validation_patient = df["patient_id"].astype(str).unique()

    validation_patient = validation_patient[0] if len(validation_patient) == 1 else ",".join(validation_patient)



    best_epoch = np.nan

    if log_path.exists():

        log = pd.read_csv(log_path)

        if "val_balanced_accuracy" in log.columns:

            best_epoch = int(log.loc[log["val_balanced_accuracy"].idxmax(), "epoch"])



    rows.append({

        "fold": run_dir.name,

        "validation_patient": validation_patient,

        "best_epoch": best_epoch,

        "n_val_windows": len(df),

        "n_awake": int((y == 0).sum()),

        "n_sleep": int((y == 1).sum()),

        "accuracy": accuracy_score(y, pred),

        "balanced_accuracy": balanced_accuracy_score(y, pred),

        "precision_sleep": precision_score(y, pred, zero_division=0),

        "recall_sensitivity_sleep": recall_score(y, pred, zero_division=0),

        "specificity_awake": tn / max(tn + fp, 1),

        "f1_score": f1_score(y, pred, zero_division=0),

        "mcc": matthews_corrcoef(y, pred),

        "kappa": cohen_kappa_score(y, pred),

        "auroc": auroc,

        "auprc": auprc,

        "tn_awake_correct": int(tn),

        "fp_awake_pred_sleep": int(fp),

        "fn_sleep_pred_awake": int(fn),

        "tp_sleep_correct": int(tp),

    })



res = pd.DataFrame(rows)

out_csv = LOPO_DIR / "lopo_summary_metrics.csv"

res.to_csv(out_csv, index=False)



print("\nPer-fold LOPO metrics:")

print(res.to_string(index=False))



metric_cols = [

    "accuracy", "balanced_accuracy", "precision_sleep",

    "recall_sensitivity_sleep", "specificity_awake",

    "f1_score", "mcc", "kappa", "auroc", "auprc"

]



summary = []

for c in metric_cols:

    summary.append({

        "metric": c,

        "mean": res[c].mean(),

        "std": res[c].std(),

        "min": res[c].min(),

        "max": res[c].max(),

    })



summary_df = pd.DataFrame(summary)

summary_csv = LOPO_DIR / "lopo_mean_std_metrics.csv"

summary_df.to_csv(summary_csv, index=False)



print("\nMean ± std summary:")

for _, r in summary_df.iterrows():

    print(f"{r['metric']}: {r['mean']:.4f} ± {r['std']:.4f}")



print("\nSaved:")

print(out_csv)

print(summary_csv)

