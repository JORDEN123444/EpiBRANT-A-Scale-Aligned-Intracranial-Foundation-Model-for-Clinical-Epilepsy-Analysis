
#!/usr/bin/env bash

set -e



CODE_DIR="$(pwd)"



PROC_ROOT="/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection/processed_patients"

BASE_ROOT="/home/ubuntu/IJAZ/SEEG datast/finetunedata/sezuire detetection"

RUN_ROOT="$BASE_ROOT/runs_lopo_patientwise_5s_pretrained"

PRETRAINED="/home/ubuntu/IJAZ/SEEG datast/pretrainingdataset/runs/scale_aligned_brant_edf_stream_real/final_pretrained.pt"



mkdir -p "$RUN_ROOT"

mkdir -p downstream/seizure_detection/configs



patients=(Patient_1 Patient_2 Patient_3 Patient_4 Patient_5 Patient_6)



for i in "${!patients[@]}"; do

    TEST_PATIENT="${patients[$i]}"

    VAL_INDEX=$(( (i + 1) % 6 ))

    VAL_PATIENT="${patients[$VAL_INDEX]}"



    TRAIN_PATIENTS=()

    for p in "${patients[@]}"; do

        if [[ "$p" != "$TEST_PATIENT" && "$p" != "$VAL_PATIENT" ]]; then

            TRAIN_PATIENTS+=("$p")

        fi

    done



    FOLD_NAME="fold_test_${TEST_PATIENT}_val_${VAL_PATIENT}"

    MANIFEST="$BASE_ROOT/lopo_5s_${FOLD_NAME}.csv"

    CONFIG="downstream/seizure_detection/configs/lopo_5s_${FOLD_NAME}.yaml"

    OUT_DIR="$RUN_ROOT/$FOLD_NAME"



    echo "========================================================"

    echo "Fold: $FOLD_NAME"

    echo "Train: ${TRAIN_PATIENTS[*]}"

    echo "Val:   $VAL_PATIENT"

    echo "Test:  $TEST_PATIENT"

    echo "========================================================"



    python downstream/seizure_detection/build_patientwise_manifest_customsplit.py \

        --processed_root "$PROC_ROOT" \

        --out_csv "$MANIFEST" \

        --window_sec 5 \

        --stride_sec 2 \

        --max_windows_per_file 1000 \

        --train_patients "${TRAIN_PATIENTS[@]}" \

        --val_patients "$VAL_PATIENT" \

        --test_patients "$TEST_PATIENT" \

        --balance_splits



    cat > "$CONFIG" <<YAML

data:

  manifest_csv: "$MANIFEST"

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

  out_dir: "$OUT_DIR"

YAML



    PYTHONPATH="$CODE_DIR:$PYTHONPATH" python downstream/seizure_detection/train_seizure.py \

        --config "$CONFIG" \

        --pretrained "$PRETRAINED" \

        --tag pretrained_full_finetune



    PYTHONPATH="$CODE_DIR:$PYTHONPATH" python downstream/seizure_detection/threshold_metrics.py \

        --run_dir "$OUT_DIR/pretrained_full_finetune"



done



echo "All LOPO folds completed."

