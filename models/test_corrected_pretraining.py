import os
import sys


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )


# ============================================================
# GPU SELECTION
#
# IMPORTANT:
# GPU selection MUST happen before torch import.
# ============================================================

from utils.gpu_selector import set_best_gpu


physical_gpu = set_best_gpu(
    min_free_memory_gb=45.0,
    max_utilization=40,
)


# ============================================================
# IMPORTS
# ============================================================

import torch
import pandas as pd

from datasets.epibrant_pretrain_dataset import (
    EpiBRANTPretrainDataset
)

from models.epibrant500m_pretrain import (
    EpiBRANT500M
)

from losses.pretrain_loss import (
    masked_reconstruction_loss
)


# ============================================================
# CUDA CHECK
# ============================================================

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is not available."
    )


device = torch.device(
    "cuda:0"
)


torch.set_float32_matmul_precision(
    "high"
)


print()
print("=" * 70)
print("GPU SELECTION")
print("=" * 70)

print(
    "Physical GPU selected:",
    physical_gpu
)

print(
    "Logical CUDA device:",
    torch.cuda.current_device()
)

print(
    "GPU:",
    torch.cuda.get_device_name(0)
)

print(
    "Total memory:",
    f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
)

print(
    "BF16 supported:",
    torch.cuda.is_bf16_supported()
)

print("=" * 70)


if not torch.cuda.is_bf16_supported():
    raise RuntimeError(
        "Selected GPU does not support BF16."
    )


# ============================================================
# MANIFEST
# ============================================================

MANIFEST = os.path.join(
    PROJECT_ROOT,
    "global_pretrain_manifest.csv"
)


if not os.path.exists(MANIFEST):
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST}"
    )


manifest_df = pd.read_csv(
    MANIFEST
)


required_manifest_columns = [
    "dataset",
    "subject_id",
    "sampling_rate",
    "file",
]


for column in required_manifest_columns:

    if column not in manifest_df.columns:
        raise RuntimeError(
            f"Manifest missing required column: {column}"
        )


print()
print("=" * 70)
print("GLOBAL MANIFEST")
print("=" * 70)

print(
    "Total manifest recordings:",
    len(manifest_df)
)

print()
print(
    "Dataset counts:"
)

print(
    manifest_df[
        "dataset"
    ].value_counts()
)

print()
print(
    "Unique subjects by source:"
)

print(
    manifest_df.groupby(
        "dataset"
    )["subject_id"].nunique()
)


# ============================================================
# DATASET LOADER
# ============================================================

dataset = EpiBRANTPretrainDataset(
    MANIFEST
)


print()
print("=" * 70)
print("DATASET LOADER")
print("=" * 70)

print(
    "Dataset recordings:",
    len(dataset)
)


if len(dataset) != len(manifest_df):

    print(
        "WARNING:"
    )

    print(
        "Dataset length differs from manifest length."
    )

    print(
        "Manifest rows:",
        len(manifest_df)
    )

    print(
        "Dataset rows:",
        len(dataset)
    )


# ============================================================
# LOCATE ONE SAMPLE FROM EACH DATASET
# ============================================================

required_sources = [
    "Institutional_SEEG",
    "Omni_iEEG_SEEG",
]


source_indices = {}


for source in required_sources:

    matches = manifest_df.index[
        manifest_df["dataset"].astype(str)
        == source
    ].tolist()

    if len(matches) == 0:

        raise RuntimeError(
            f"No recording found for dataset source: {source}"
        )

    source_indices[source] = matches[0]


print()
print("=" * 70)
print("TEST SAMPLE INDICES")
print("=" * 70)

for source, index in source_indices.items():

    print(
        f"{source}: row {index}"
    )


# ============================================================
# CHECK DATASET SAMPLE
# ============================================================

def prepare_sample(
    sample_index,
    expected_source,
):

    sample = dataset[
        sample_index
    ]

    print()
    print("=" * 70)
    print(
        f"REAL SAMPLE: {expected_source}"
    )
    print("=" * 70)

    print(
        "Returned keys:",
        list(sample.keys())
    )


    # --------------------------------------------------------
    # Required loader outputs
    # --------------------------------------------------------

    required_keys = [
        "signal",
        "fs",
        "channel_mask",
        "dataset",
        "subject",
    ]


    for key in required_keys:

        if key not in sample:

            raise RuntimeError(
                f"\nDataset loader is missing required key: '{key}'\n"
                f"Returned keys: {list(sample.keys())}\n\n"
                "Update datasets/epibrant_pretrain_dataset.py "
                "before starting pretraining."
            )


    signal = sample[
        "signal"
    ]

    fs = sample[
        "fs"
    ]

    channel_mask = sample[
        "channel_mask"
    ].bool()

    source = str(
        sample[
            "dataset"
        ]
    )


    # --------------------------------------------------------
    # Validate data source metadata
    # --------------------------------------------------------

    if source != expected_source:

        raise RuntimeError(
            f"Dataset-source mismatch. "
            f"Expected '{expected_source}' "
            f"but loader returned '{source}'."
        )


    # --------------------------------------------------------
    # Validate signal
    # --------------------------------------------------------

    if signal.ndim != 2:

        raise RuntimeError(
            f"Expected signal [C,T], "
            f"got {tuple(signal.shape)}"
        )


    C, T = signal.shape


    # --------------------------------------------------------
    # Native sampling rate
    # --------------------------------------------------------

    fs_float = float(
        fs
    )


    expected_samples = int(
        round(
            fs_float * 60.0
        )
    )


    if T != expected_samples:

        raise RuntimeError(
            f"Expected {expected_samples} samples "
            f"for 60 s at fs={fs_float}, "
            f"but received {T}."
        )


    # --------------------------------------------------------
    # Channel mask
    # --------------------------------------------------------

    if channel_mask.ndim != 1:

        raise RuntimeError(
            f"channel_mask must be [C], "
            f"got {tuple(channel_mask.shape)}"
        )


    if channel_mask.shape[0] != C:

        raise RuntimeError(
            f"channel_mask length "
            f"{channel_mask.shape[0]} "
            f"does not match signal channels {C}."
        )


    num_valid = int(
        channel_mask.sum().item()
    )

    num_padded = (
        C - num_valid
    )


    if num_valid <= 0:

        raise RuntimeError(
            "Recording contains zero valid SEEG channels."
        )


    print(
        "Signal:",
        signal.shape
    )

    print(
        "Sampling rate:",
        fs_float
    )

    print(
        "Dataset:",
        source
    )

    print(
        "Subject:",
        sample[
            "subject"
        ]
    )

    print(
        "Session:",
        sample.get(
            "session",
            ""
        )
    )

    print(
        "Recording:",
        sample.get(
            "recording",
            ""
        )
    )

    print(
        "Valid SEEG channels:",
        num_valid
    )

    print(
        "Padded channels:",
        num_padded
    )

    print(
        "Total returned channels:",
        C
    )


    if "num_valid_channels" in sample:

        reported_valid = int(
            sample[
                "num_valid_channels"
            ]
        )

        if reported_valid != num_valid:

            raise RuntimeError(
                f"num_valid_channels={reported_valid}, "
                f"but channel_mask contains "
                f"{num_valid} valid channels."
            )


    # --------------------------------------------------------
    # Move to GPU
    # --------------------------------------------------------

    signal = (
        signal
        .unsqueeze(0)
        .to(
            device=device,
            dtype=torch.float32
        )
    )


    channel_mask = (
        channel_mask
        .unsqueeze(0)
        .to(
            device=device
        )
    )


    print(
        "Model input:",
        signal.shape
    )

    print(
        "Channel mask:",
        channel_mask.shape
    )


    return {
        "signal":
            signal,

        "fs":
            fs_float,

        "channel_mask":
            channel_mask,

        "dataset":
            source,

        "subject":
            sample["subject"],
    }


# ============================================================
# LOAD REAL SAMPLES FROM BOTH SOURCES
# ============================================================

institutional_sample = prepare_sample(
    source_indices[
        "Institutional_SEEG"
    ],
    "Institutional_SEEG",
)


omni_sample = prepare_sample(
    source_indices[
        "Omni_iEEG_SEEG"
    ],
    "Omni_iEEG_SEEG",
)


# ============================================================
# BUILD EpiBRANT-500M
#
# IMPORTANT:
# Do NOT override temporal/channel/decoder depths here.
#
# epibrant500m_pretrain.py should contain:
#
# temporal_depth = 20
# channel_depth  = 12
# decoder_depth  = 7
# ============================================================

print()
print("=" * 70)
print("BUILDING EpiBRANT-500M")
print("=" * 70)


model = EpiBRANT500M().to(
    device
)


model.train()


# ============================================================
# PARAMETER COUNT
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)


trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print(
    f"Total parameters: "
    f"{total_params:,}"
)

print(
    f"Total parameters: "
    f"{total_params / 1e6:.3f} M"
)

print(
    f"Trainable parameters: "
    f"{trainable_params / 1e6:.3f} M"
)


# ============================================================
# REQUIRE APPROXIMATELY 500M
# ============================================================

MIN_EXPECTED_PARAMS = 495_000_000

MAX_EXPECTED_PARAMS = 505_000_000


if not (
    MIN_EXPECTED_PARAMS
    <= total_params
    <= MAX_EXPECTED_PARAMS
):

    raise RuntimeError(
        "\nEpiBRANT parameter count is not approximately 500M.\n"
        f"Current parameter count: {total_params:,}\n"
        f"Current size: {total_params / 1e6:.3f} M\n"
        "Expected range: 495M505M.\n\n"
        "Check models/epibrant500m_pretrain.py defaults:\n"
        "temporal_depth=20\n"
        "channel_depth=12\n"
        "decoder_depth=7"
    )


print(
    "EpiBRANT-500M parameter check: SUCCESS"
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=1e-5,
    betas=(
        0.9,
        0.95
    ),
    weight_decay=0.01,
)


# ============================================================
# REAL FORWARD/BACKWARD TEST
# ============================================================

def run_training_step(
    batch,
    step_name,
):

    print()
    print("=" * 70)
    print(
        f"TRAINING TEST: {step_name}"
    )
    print("=" * 70)


    optimizer.zero_grad(
        set_to_none=True
    )


    torch.cuda.empty_cache()

    torch.cuda.reset_peak_memory_stats()


    # ========================================================
    # BF16 FORWARD
    # ========================================================

    with torch.autocast(
        device_type="cuda",
        dtype=torch.bfloat16,
    ):

        output = model(
            batch[
                "signal"
            ],
            batch[
                "fs"
            ],
            channel_mask=batch[
                "channel_mask"
            ],
        )


        loss = masked_reconstruction_loss(
            prediction=output[
                "reconstruction"
            ],
            target=output[
                "target"
            ],
            mask=output[
                "mask"
            ],
            channel_mask=output[
                "channel_mask"
            ],
            loss_type="smooth_l1",
        )


    # ========================================================
    # OUTPUT VALIDATION
    # ========================================================

    required_output_keys = [
        "tokens",
        "target",
        "masked_tokens",
        "latent",
        "reconstruction",
        "mask",
        "channel_mask",
    ]


    for key in required_output_keys:

        if key not in output:

            raise RuntimeError(
                f"Missing model output: {key}"
            )


    print(
        "Tokens:",
        output[
            "tokens"
        ].shape
    )

    print(
        "Masked tokens:",
        output[
            "masked_tokens"
        ].shape
    )

    print(
        "Latent:",
        output[
            "latent"
        ].shape
    )

    print(
        "Reconstruction:",
        output[
            "reconstruction"
        ].shape
    )

    print(
        "Mask:",
        output[
            "mask"
        ].shape
    )


    # ========================================================
    # MASK RATIO  VALID CONTACTS ONLY
    # ========================================================

    valid_token_mask = (
        output[
            "channel_mask"
        ]
        .unsqueeze(-1)
        .expand_as(
            output[
                "mask"
            ]
        )
    )


    valid_masks = (
        output[
            "mask"
        ][
            valid_token_mask
        ]
    )


    if valid_masks.numel() == 0:

        raise RuntimeError(
            "No valid SEEG tokens found."
        )


    mask_ratio = (
        valid_masks
        .float()
        .mean()
        .item()
    )


    print(
        "Mask ratio over valid SEEG contacts:",
        mask_ratio
    )


    if not (
        0.30
        <= mask_ratio
        <= 0.50
    ):

        raise RuntimeError(
            f"Unexpected valid-channel "
            f"mask ratio: {mask_ratio}"
        )


    # ========================================================
    # ENSURE PADDING IS NEVER MASKED
    # ========================================================

    padding_token_mask = (
        ~output[
            "channel_mask"
        ]
    ).unsqueeze(
        -1
    ).expand_as(
        output[
            "mask"
        ]
    )


    if padding_token_mask.any():

        padded_selected = (
            output[
                "mask"
            ][
                padding_token_mask
            ]
            .any()
            .item()
        )


        if padded_selected:

            raise RuntimeError(
                "ERROR: padded channels were selected "
                "for masked reconstruction."
            )


    print(
        "Padding exclusion check: SUCCESS"
    )


    # ========================================================
    # FINITE LOSS
    # ========================================================

    print(
        "Loss:",
        float(
            loss.item()
        )
    )


    if not torch.isfinite(
        loss
    ):

        raise RuntimeError(
            "Loss is NaN or Inf."
        )


    # ========================================================
    # BACKWARD
    # ========================================================

    print(
        "Running backward..."
    )


    loss.backward()


    # ========================================================
    # GRADIENT CHECK
    # ========================================================

    grad_norm = (
        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )
    )


    grad_norm_value = float(
        grad_norm
    )


    print(
        "Gradient norm:",
        grad_norm_value
    )


    if not torch.isfinite(
        torch.as_tensor(
            grad_norm_value
        )
    ):

        raise RuntimeError(
            "Gradient norm is NaN or Inf."
        )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    optimizer.step()


    print(
        "Optimizer step: SUCCESS"
    )


    # ========================================================
    # GPU MEMORY
    # ========================================================

    allocated = (
        torch.cuda.memory_allocated()
        / 1024**3
    )

    reserved = (
        torch.cuda.memory_reserved()
        / 1024**3
    )

    peak = (
        torch.cuda.max_memory_allocated()
        / 1024**3
    )


    print(
        f"GPU allocated: "
        f"{allocated:.2f} GB"
    )

    print(
        f"GPU reserved: "
        f"{reserved:.2f} GB"
    )

    print(
        f"GPU peak allocated: "
        f"{peak:.2f} GB"
    )


    print(
        f"{step_name}: SUCCESS"
    )


# ============================================================
# TEST INSTITUTIONAL DATA
# ============================================================

run_training_step(
    institutional_sample,
    "Institutional_SEEG",
)


# ============================================================
# TEST OMNI DATA
# ============================================================

run_training_step(
    omni_sample,
    "Omni_iEEG_SEEG",
)


# ============================================================
# FINAL RESULT
# ============================================================

print()
print("=" * 70)
print(
    "EpiBRANT-500M CORRECTED PRETRAINING TEST: SUCCESS"
)
print("=" * 70)

print(
    "Verified:"
)

print(
    "  EpiBRANT H 500M parameters"
)

print(
    "  Institutional SEEG loading"
)

print(
    "  Omni-iEEG loading"
)

print(
    "  Native sampling frequency"
)

print(
    "  Real/padded channel masking"
)

print(
    "  ~40% valid-token masking"
)

print(
    "  BF16 forward"
)

print(
    "  Masked reconstruction loss"
)

print(
    "  Backward propagation"
)

print(
    "  Gradient clipping"
)

print(
    "  AdamW optimizer step"
)

print("=" * 70)