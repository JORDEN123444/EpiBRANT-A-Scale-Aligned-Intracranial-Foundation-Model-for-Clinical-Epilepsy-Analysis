import os
import subprocess


def get_gpu_status():
    """
    Return status of all physical NVIDIA GPUs.
    """

    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=True,
    )

    gpus = []

    for line in result.stdout.strip().splitlines():

        if not line.strip():
            continue

        parts = [p.strip() for p in line.split(",")]

        if len(parts) != 6:
            continue

        gpus.append(
            {
                "index": int(parts[0]),
                "name": parts[1],
                "total_memory_mb": int(parts[2]),
                "used_memory_mb": int(parts[3]),
                "free_memory_mb": int(parts[4]),
                "utilization_percent": int(parts[5]),
            }
        )

    if not gpus:
        raise RuntimeError(
            "No NVIDIA GPUs detected."
        )

    return gpus


def print_gpu_status():

    gpus = get_gpu_status()

    print("=" * 70)
    print("NVIDIA GPU STATUS")
    print("=" * 70)

    for gpu in gpus:

        print(
            f"GPU {gpu['index']}: "
            f"{gpu['name']}"
        )

        print(
            f"  Free memory : "
            f"{gpu['free_memory_mb'] / 1024:.2f} GB"
        )

        print(
            f"  Used memory : "
            f"{gpu['used_memory_mb'] / 1024:.2f} GB"
        )

        print(
            f"  Utilization : "
            f"{gpu['utilization_percent']}%"
        )

        print("-" * 70)


def select_free_gpu(
    min_free_memory_gb=20.0,
    max_utilization=20,
):
    """
    Select the best currently available physical GPU.

    Selection:
        - enough free VRAM
        - low GPU utilization
        - lowest utilization first
        - highest free memory second
    """

    gpus = get_gpu_status()

    candidates = []

    for gpu in gpus:

        free_gb = (
            gpu["free_memory_mb"] / 1024.0
        )

        utilization = (
            gpu["utilization_percent"]
        )

        if (
            free_gb >= min_free_memory_gb
            and
            utilization <= max_utilization
        ):

            candidates.append(gpu)

    if not candidates:

        print_gpu_status()

        raise RuntimeError(
            "\nNo suitable free GPU found.\n"
            f"Required free memory: "
            f"{min_free_memory_gb:.1f} GB\n"
            f"Maximum utilization: "
            f"{max_utilization}%"
        )

    candidates.sort(
        key=lambda gpu: (
            gpu["utilization_percent"],
            -gpu["free_memory_mb"],
        )
    )

    return candidates[0]["index"]


def set_best_gpu(
    min_free_memory_gb=20.0,
    max_utilization=20,
):
    """
    Automatically select one physical GPU.

    This should be called BEFORE importing torch.

    If CUDA_VISIBLE_DEVICES is already set,
    that selection is respected.
    """

    existing = os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    )

    if existing:
        print(
            "CUDA_VISIBLE_DEVICES already set:",
            existing
        )
        return existing

    gpu_index = select_free_gpu(
        min_free_memory_gb=min_free_memory_gb,
        max_utilization=max_utilization,
    )

    os.environ["CUDA_VISIBLE_DEVICES"] = str(
        gpu_index
    )

    print("=" * 70)
    print("AUTOMATIC GPU SELECTION")
    print("=" * 70)

    print(
        "Selected physical GPU:",
        gpu_index
    )

    print(
        "CUDA_VISIBLE_DEVICES:",
        os.environ["CUDA_VISIBLE_DEVICES"]
    )

    print("=" * 70)

    return str(gpu_index)


if __name__ == "__main__":

    print_gpu_status()

    print()

    try:

        selected = select_free_gpu(
            min_free_memory_gb=20.0,
            max_utilization=20,
        )

        print(
            "Selected GPU:",
            selected
        )

    except RuntimeError as e:

        print(e)
