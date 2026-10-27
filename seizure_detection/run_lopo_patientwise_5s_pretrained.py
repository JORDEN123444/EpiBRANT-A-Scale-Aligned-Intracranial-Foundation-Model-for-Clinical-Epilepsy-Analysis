
import os

import sys

import subprocess

from pathlib import Path

import textwrap



CODE_DIR = Path.cwd()



PROC_ROOT = "/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection/processed_patients"

BASE_ROOT = "/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection"

RUN_ROOT = f"{BASE_ROOT}/runs_lopo_patientwise_5s_pretrained"

PRETRAINED = "/home/ubuntu/IJAZ/SEEG datast/pretrainingdataset/runs/scale_aligned_brant_edf_stream_real/final_pretrained.pt"



Path(RUN_ROOT).mkdir(parents=True, exist_ok=True)

Path("downstream/seizure_detection/configs").mkdir(parents=True, exist_ok=True)



patients = ["Patient_1", "Patient_2", "Patient_3", "Patient_4", "Patient_5", "Patient_6"]



env = os.environ.copy()

env["PYTHONPATH"] = f"{CODE_DIR}:{env.get('PYTHONPATH', '')}"



for i, test_patient in enumerate(patients):

    val_patient = patients[(i + 1) % len(patients)]

    train_patients = [p for p in patients if p not in [test_patient, val_patient]]



    fold_name = f"fold_test_{test_patient}_val_{val_patient}"

    manifest = f"{BASE_ROOT}/lopo_5s_{fold_name}.csv"

    config = f"downstream/seizure_detection/configs/lopo_5s_{fold_name}.yaml"

    out_dir = f"{RUN_ROOT}/{fold_name}"



    print("=" * 70)

    print("Fold:", fold_name)

    print("Train:", train_patients)

    print("Val:  ", val_patient)

    print("Test: ", test_patient)

    print("=" * 70)



    metric_done = Path(out_dir) / "pretrained_full_finetune" / "test_metrics_threshold_optimized.csv"

    if metric_done.exists():

        print("[SKIP] Fold already completed:", metric_done)

        continue



    cmd_manifest = [

        sys.executable,

        "downstream/seizure_detection/build_patientwise_manifest_customsplit.py",

        "--processed_root", PROC_ROOT,

        "--out_csv", manifest,

        "--window_sec", "5",

        "--stride_sec", "2",

        "--max_windows_per_file", "1000",

        "--train_patients", *train_patients,

        "--val_patients", val_patient,

        "--test_patients", test_patient,

        "--balance_splits",

    ]



    subprocess.run(cmd_manifest, check=True, env=env)



    yaml_text = f"""

data:

  manifest_csv: "{manifest}"

  context_sec: 5

  patch_sec: 1

  max_channels: 128



model:

  d_model: 192

  temporal_layers: 3

  channel_layers: 2

  n_heads: 6

  dim_feedforward: 768

  dropout: 0.1

  t_align: 8

  f_align: 128

  f_max: 128.0

  conv_channels: 24



train:

  epochs: 30

  batch_size: 1

  backbone_lr: 1.0e-6

  head_lr: 5.0e-5

  weight_decay: 0.001

  grad_clip: 1.0

  amp: true

  num_workers: 0

  seed: 42

  out_dir: "{out_dir}"

"""

    Path(config).write_text(textwrap.dedent(yaml_text).strip() + "\n")



    cmd_train = [

        sys.executable,

        "downstream/seizure_detection/train_seizure.py",

        "--config", config,

        "--pretrained", PRETRAINED,

        "--tag", "pretrained_full_finetune",

    ]



    subprocess.run(cmd_train, check=True, env=env)



    cmd_threshold = [

        sys.executable,

        "downstream/seizure_detection/threshold_metrics.py",

        "--run_dir", f"{out_dir}/pretrained_full_finetune",

    ]



    subprocess.run(cmd_threshold, check=True, env=env)



print("\nAll LOPO folds completed.")

