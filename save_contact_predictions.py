import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import numpy as np
import pandas as pd


# =========================================================
# PATHS
# =========================================================

ROOT = Path("downstream/soz_localization")
EMB_DIR = ROOT / "embeddings"
CKPT_DIR = ROOT / "training" / "checkpoints"
MANIFEST_PATH = ROOT / "manifests" / "soz_manifest.csv"
OUT_DIR = ROOT / "evaluation" / "results"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# MODEL
# =========================================================

class SOZHead(nn.Module):
    def __init__(self, dim=1024):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fc = nn.Linear(dim, 1)

    def forward(self, x):
        x = self.norm(x)
        x = self.fc(x)
        return x.squeeze(-1)


def load_model(ckpt_path):
    model = SOZHead(dim=1024)
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()
    return model


# =========================================================
# HELPERS
# =========================================================

AUXILIARY_KEYWORDS = [
    "ECG", "EKG", "EMG", "EOG", "RESP", "RESPIR",
    "SPO2", "SP02", "PLETH", "PULSE", "STATUS",
    "TRIG", "TRIGGER", "EVENT", "ANNOT", "MARK"
]


def is_auxiliary_channel(name):
    name = str(name).upper()
    return any(k in name for k in AUXILIARY_KEYWORDS)


def recover_channel_table(channels_file, expected_n):
    df = pd.read_csv(channels_file, sep="\t")

    # try SEEG-only first
    if "type" in df.columns:
        seeg_df = df[df["type"].astype(str).str.upper().str.contains("SEEG", na=False)].copy()
        if len(seeg_df) == expected_n:
            return seeg_df.reset_index(drop=True)

    # fallback: remove auxiliary channels
    if "name" in df.columns:
        non_aux = df[~df["name"].astype(str).apply(is_auxiliary_channel)].copy()
        if len(non_aux) == expected_n:
            return non_aux.reset_index(drop=True)

    # fallback: first expected_n rows
    if len(df) >= expected_n:
        return df.iloc[:expected_n].reset_index(drop=True)

    raise RuntimeError(
        f"Could not recover {expected_n} channels from {channels_file}. Found only {len(df)} rows."
    )


def short_patient_name(pid):
    pid = pid.replace("sub-", "")
    pid = pid.replace("openieeg", "")
    return pid


# =========================================================
# MAIN
# =========================================================

manifest = pd.read_csv(MANIFEST_PATH)

patients = sorted([p.stem for p in EMB_DIR.glob("*.pt")])

all_rows = []
patient_summary = []

for fold_idx, patient in enumerate(patients, start=1):
    print("=" * 80)
    print("Patient:", patient)

    emb_path = EMB_DIR / f"{patient}.pt"
    ckpt_path = CKPT_DIR / f"fold_{fold_idx}.pt"

    if not ckpt_path.exists():
        print("Missing checkpoint:", ckpt_path)
        continue

    data = torch.load(emb_path, map_location="cpu")

    embedding = data["embedding"]          # [W,288,1024]
    soz_label = data["soz_label"]          # [288]
    channel_mask = data["channel_mask"]    # [288]
    patient_id = data["patient_id"]

    # average over windows -> [288,1024]
    embedding = embedding.mean(dim=0)

    valid = channel_mask.bool()
    embedding = embedding[valid]
    soz_label = soz_label[valid]

    n_valid = int(valid.sum().item())

    # channel file from manifest
    sub_df = manifest[manifest["patient_id"] == patient]
    if len(sub_df) == 0:
        raise RuntimeError(f"No manifest rows found for patient {patient}")

    channels_file = sub_df.iloc[0]["channels_file"]
    channel_df = recover_channel_table(channels_file, n_valid)

    model = load_model(ckpt_path)

    with torch.no_grad():
        logits = model(embedding)
        prob = torch.sigmoid(logits).numpy()

    y = soz_label.numpy().astype(int)
    pred_label = (prob >= 0.5).astype(int)

    # descending rank
    order = np.argsort(-prob)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, len(prob) + 1)

    # summary
    tp = int(((pred_label == 1) & (y == 1)).sum())
    tn = int(((pred_label == 0) & (y == 0)).sum())
    fp = int(((pred_label == 1) & (y == 0)).sum())
    fn = int(((pred_label == 0) & (y == 1)).sum())

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    precision = tp / (tp + fp + 1e-8)

    patient_summary.append({
        "patient_id": patient,
        "patient_short": short_patient_name(patient),
        "n_contacts": n_valid,
        "n_soz": int(y.sum()),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "precision": precision,
    })

    # save each contact
    for i in range(n_valid):
        row = {
            "patient_id": patient,
            "patient_short": short_patient_name(patient),
            "contact_index": i,
            "contact_name": channel_df.iloc[i]["name"] if "name" in channel_df.columns else f"ch_{i+1}",
            "anatomical": channel_df.iloc[i]["anatomical"] if "anatomical" in channel_df.columns else "",
            "true_soz": int(y[i]),
            "probability": float(prob[i]),
            "pred_label": int(pred_label[i]),
            "rank": int(ranks[i]),
            "channels_file": channels_file,
        }
        all_rows.append(row)

    print(f"Contacts: {n_valid} | SOZ: {int(y.sum())}")

contact_df = pd.DataFrame(all_rows)
summary_df = pd.DataFrame(patient_summary)

contact_csv = OUT_DIR / "contact_predictions.csv"
summary_csv = OUT_DIR / "contact_prediction_summary.csv"

contact_df.to_csv(contact_csv, index=False)
summary_df.to_csv(summary_csv, index=False)

print("\nDONE")
print("Saved:", contact_csv)
print("Saved:", summary_csv)
