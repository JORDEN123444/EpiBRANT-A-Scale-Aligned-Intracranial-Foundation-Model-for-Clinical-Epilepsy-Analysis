
import argparse

import os

import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))



import numpy as np

import torch

import yaml

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from torch.utils.data import DataLoader



from model_scale_brant import ScaleAlignedBrantMAE

from downstream.imputation.data_imputation import ImputationWindowDataset





def make_model(cfg):

    return ScaleAlignedBrantMAE(

        d_model=cfg["model"]["d_model"],

        seq_len=cfg["data"]["context_sec"] // cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        temporal_layers=cfg["model"]["temporal_layers"],

        channel_layers=cfg["model"]["channel_layers"],

        n_heads=cfg["model"]["n_heads"],

        dim_feedforward=cfg["model"]["dim_feedforward"],

        dropout=cfg["model"]["dropout"],

        mask_ratio=cfg["train"]["mask_ratio"],

        t_align=cfg["model"]["t_align"],

        f_align=cfg["model"]["f_align"],

        f_max=cfg["model"]["f_max"],

        conv_channels=cfg["model"]["conv_channels"],

    )





def stitch_channel_spectrogram(arr, channel_index):

    """

    arr shape: [B, C, L, 1, T_align, F_align]

    output: [F_align, L*T_align]

    """

    x = arr[0, channel_index, :, 0]      # [L, T_align, F_align]

    x = x.permute(2, 0, 1)              # [F_align, L, T_align]

    f, l, t = x.shape

    x = x.reshape(f, l * t)             # [F_align, total_time_bins]

    return x.detach().cpu().numpy()





def make_masked_input(original, mask_pos, channel_index, t_align):

    """

    original: [F, L*T_align]

    mask_pos: [B, C, L]

    """

    masked = original.copy()

    mask_line = mask_pos[0, channel_index].detach().cpu().numpy().astype(bool)



    for i, m in enumerate(mask_line):

        if m:

            masked[:, i * t_align:(i + 1) * t_align] = np.nan



    return masked





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument("--checkpoint", required=True)

    parser.add_argument("--out_dir", required=True)

    parser.add_argument("--tag", default="pretrained_sabrant")

    args = parser.parse_args()



    os.makedirs(args.out_dir, exist_ok=True)



    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"



    test_ds = ImputationWindowDataset(

        manifest_csv=cfg["data"]["manifest_csv"],

        split="test",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        windows_per_run=1,

        seed=cfg["train"]["seed"] + 5000,

    )



    loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0)



    model = make_model(cfg)

    state = torch.load(args.checkpoint, map_location="cpu")

    missing, unexpected = model.load_state_dict(state, strict=False)



    print("Loaded checkpoint:", args.checkpoint)

    print("Missing:", len(missing))

    print("Unexpected:", len(unexpected))



    model = model.to(device)

    model.eval()



    batch = next(iter(loader))



    x = batch["x"].to(device)

    fs = batch["fs"].to(device)

    channel_mask = batch["channel_mask"].to(device)



    with torch.no_grad():

        with torch.cuda.amp.autocast(enabled=amp_enabled):

            pred, target, mask_pos = model(x, fs, channel_mask)



    # choose a real channel with many masked patches

    real_channels = channel_mask[0].detach().cpu().numpy().astype(bool)

    mask_count = mask_pos[0].detach().cpu().numpy().sum(axis=1)

    mask_count[~real_channels] = -1

    ch = int(np.argmax(mask_count))



    print("Selected channel index:", ch)

    print("Masked patches in selected channel:", int(mask_count[ch]))



    t_align = int(cfg["model"]["t_align"])



    original = stitch_channel_spectrogram(target, ch)

    reconstructed = stitch_channel_spectrogram(pred, ch)

    masked_input = make_masked_input(original, mask_pos, ch, t_align)

    absolute_error = np.abs(original - reconstructed)



    # save arrays

    npz_path = os.path.join(args.out_dir, f"imputation_reconstruction_arrays_{args.tag}.npz")

    np.savez(

        npz_path,

        original_spectrogram=original,

        masked_input=masked_input,

        reconstructed_spectrogram=reconstructed,

        absolute_error=absolute_error,

        selected_channel_index=ch,

    )



    print("Saved arrays:", npz_path)



    # common intensity range for original/masked/reconstructed

    common_values = np.concatenate([

        original[np.isfinite(original)].reshape(-1),

        reconstructed[np.isfinite(reconstructed)].reshape(-1),

    ])



    vmin = np.percentile(common_values, 1)

    vmax = np.percentile(common_values, 99)



    err_vmax = np.percentile(absolute_error[np.isfinite(absolute_error)], 99)



    plt.rcParams.update({

        "font.family": "sans-serif",

        "font.size": 7,

        "axes.labelsize": 7,

        "xtick.labelsize": 6,

        "ytick.labelsize": 6,

        "axes.linewidth": 0.7,

        "pdf.fonttype": 42,

        "ps.fonttype": 42,

    })



    cmap_main = plt.cm.viridis.copy()

    cmap_main.set_bad(color="lightgray")



    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.05))



    panels = [

        ("Original", original, vmin, vmax, cmap_main),

        ("Masked input", masked_input, vmin, vmax, cmap_main),

        ("Reconstructed", reconstructed, vmin, vmax, cmap_main),

        ("Absolute error", absolute_error, 0, err_vmax, plt.cm.magma),

    ]



    letters = ["a", "b", "c", "d"]



    for ax, (label, arr, lo, hi, cmap), letter in zip(axes, panels, letters):

        ax.imshow(arr, aspect="auto", origin="lower", vmin=lo, vmax=hi, cmap=cmap)

        ax.set_xlabel("Time bin")

        ax.set_ylabel("Frequency bin")

        ax.set_title(label, fontsize=7)

        ax.tick_params(axis="both", direction="out", length=2.5, width=0.7)

        ax.text(-0.22, 1.10, letter, transform=ax.transAxes,

                fontsize=9, fontweight="bold", va="top")



    plt.tight_layout(w_pad=0.7)



    for ext in ["png", "tiff", "pdf"]:

        out_path = os.path.join(args.out_dir, f"fig_nature_imputation_reconstruction_{args.tag}.{ext}")

        if ext == "pdf":

            plt.savefig(out_path, bbox_inches="tight")

        else:

            plt.savefig(out_path, dpi=600, bbox_inches="tight")

        print("Saved:", out_path)



    plt.close()





if __name__ == "__main__":

    main()

