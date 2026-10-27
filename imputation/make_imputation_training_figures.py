
import os

import pandas as pd

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt





base = "/home/ubuntu/IJAZ/SEEG datast/finetunedata/imputation/runs"

fig_dir = os.path.join(base, "nature_figures")

os.makedirs(fig_dir, exist_ok=True)



random_log = os.path.join(base, "random_init", "finetune_log.csv")

pre_log = os.path.join(base, "pretrained_sabrant", "finetune_log.csv")



logs = {}



if os.path.exists(random_log):

    logs["Random initialization"] = pd.read_csv(random_log)



if os.path.exists(pre_log):

    logs["SA-Brant-SEEG"] = pd.read_csv(pre_log)



if len(logs) == 0:

    raise FileNotFoundError("No finetune_log.csv found. Finish training first.")



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





# -----------------------------

# Figure 1: Fine-tuning curves

# -----------------------------

fig, ax = plt.subplots(figsize=(3.4, 2.5))



for name, df in logs.items():

    if "epoch" in df.columns and "val_loss" in df.columns:

        ax.plot(df["epoch"], df["val_loss"], linewidth=1.0, label=name)



ax.set_xlabel("Fine-tuning epoch")

ax.set_ylabel("Validation reconstruction loss")

ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", direction="out", length=2.5, width=0.7)

ax.legend(frameon=False)



plt.tight_layout()



for ext in ["png", "tiff", "pdf"]:

    path = os.path.join(fig_dir, f"fig_imputation_finetuning_loss.{ext}")

    if ext == "pdf":

        plt.savefig(path, bbox_inches="tight")

    else:

        plt.savefig(path, dpi=600, bbox_inches="tight")



plt.close()





# -----------------------------

# Figure 2: Best validation loss

# -----------------------------

names = []

best_vals = []



for name, df in logs.items():

    names.append(name)

    best_vals.append(float(df["val_loss"].min()))



fig, ax = plt.subplots(figsize=(2.6, 2.4))



ax.bar(range(len(names)), best_vals, width=0.55)

ax.set_xticks(range(len(names)))

ax.set_xticklabels(names, rotation=25, ha="right")

ax.set_ylabel("Best validation loss")



ax.spines["top"].set_visible(False)

ax.spines["right"].set_visible(False)

ax.tick_params(axis="both", direction="out", length=2.5, width=0.7)



for i, v in enumerate(best_vals):

    ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=6)



plt.tight_layout()



for ext in ["png", "tiff", "pdf"]:

    path = os.path.join(fig_dir, f"fig_imputation_best_validation_loss.{ext}")

    if ext == "pdf":

        plt.savefig(path, bbox_inches="tight")

    else:

        plt.savefig(path, dpi=600, bbox_inches="tight")



plt.close()





# -----------------------------

# Save summary table

# -----------------------------

summary = pd.DataFrame({

    "model": names,

    "best_val_loss": best_vals,

})

summary.to_csv(os.path.join(fig_dir, "imputation_validation_summary.csv"), index=False)



print("Saved training figures to:")

print(fig_dir)

print(summary)

