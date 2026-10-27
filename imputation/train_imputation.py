
import argparse

from pathlib import Path



import pandas as pd

import torch

import torch.nn.functional as F

import yaml

from torch.utils.data import DataLoader



from model_scale_brant import ScaleAlignedBrantMAE

from downstream.imputation.data_imputation import ImputationWindowDataset





def masked_smooth_l1(pred, target, mask_pos):

    mask = mask_pos[:, :, :, None, None, None].expand_as(pred)

    if mask.sum() == 0:

        return pred.mean() * 0.0

    return F.smooth_l1_loss(pred[mask], target[mask])





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





def evaluate(model, loader, device, amp_enabled):

    model.eval()

    losses = []



    with torch.no_grad():

        for batch in loader:

            x = batch["x"].to(device)

            fs = batch["fs"].to(device)

            channel_mask = batch["channel_mask"].to(device)



            with torch.cuda.amp.autocast(enabled=amp_enabled):

                pred, target, mask_pos = model(x, fs, channel_mask)

                loss = masked_smooth_l1(pred, target, mask_pos)



            losses.append(float(loss.detach().cpu()))



    model.train()

    return sum(losses) / max(1, len(losses))





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--config", required=True)

    parser.add_argument("--pretrained", default=None)

    parser.add_argument("--tag", default="run")

    args = parser.parse_args()



    with open(args.config, "r") as f:

        cfg = yaml.safe_load(f)



    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    amp_enabled = bool(cfg["train"]["amp"]) and device.type == "cuda"



    out_dir = Path(cfg["train"]["out_dir"]) / args.tag

    out_dir.mkdir(parents=True, exist_ok=True)



    train_ds = ImputationWindowDataset(

        manifest_csv=cfg["data"]["manifest_csv"],

        split="train",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        windows_per_run=cfg["data"]["train_windows_per_run"],

        seed=cfg["train"]["seed"],

    )



    val_ds = ImputationWindowDataset(

        manifest_csv=cfg["data"]["manifest_csv"],

        split="val",

        context_sec=cfg["data"]["context_sec"],

        patch_sec=cfg["data"]["patch_sec"],

        max_channels=cfg["data"]["max_channels"],

        windows_per_run=cfg["data"]["val_windows_per_run"],

        seed=cfg["train"]["seed"] + 1000,

    )



    train_loader = DataLoader(

        train_ds,

        batch_size=cfg["train"]["batch_size"],

        shuffle=True,

        num_workers=cfg["train"]["num_workers"],

    )



    val_loader = DataLoader(

        val_ds,

        batch_size=cfg["train"]["batch_size"],

        shuffle=False,

        num_workers=cfg["train"]["num_workers"],

    )



    model = make_model(cfg)



    if args.pretrained is not None:

        state = torch.load(args.pretrained, map_location="cpu")

        missing, unexpected = model.load_state_dict(state, strict=False)

        print("Loaded pretrained:", args.pretrained)

        print("Missing:", len(missing))

        print("Unexpected:", len(unexpected))



    model = model.to(device)



    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=cfg["train"]["lr"],

        weight_decay=cfg["train"]["weight_decay"],

    )



    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)



    logs = []

    best_val = 1e9



    print("Device:", device)

    print("AMP:", amp_enabled)

    print("Train windows:", len(train_ds))

    print("Val windows:", len(val_ds))

    print("Output:", out_dir)



    for epoch in range(1, cfg["train"]["epochs"] + 1):

        model.train()

        total = 0.0

        count = 0



        for batch in train_loader:

            x = batch["x"].to(device)

            fs = batch["fs"].to(device)

            channel_mask = batch["channel_mask"].to(device)



            optimizer.zero_grad(set_to_none=True)



            with torch.cuda.amp.autocast(enabled=amp_enabled):

                pred, target, mask_pos = model(x, fs, channel_mask)

                loss = masked_smooth_l1(pred, target, mask_pos)



            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["train"]["grad_clip"])

            scaler.step(optimizer)

            scaler.update()



            total += float(loss.detach().cpu())

            count += 1



        train_loss = total / max(1, count)

        val_loss = evaluate(model, val_loader, device, amp_enabled)



        row = {

            "epoch": epoch,

            "train_loss": train_loss,

            "val_loss": val_loss,

        }

        logs.append(row)



        print(row)



        pd.DataFrame(logs).to_csv(out_dir / "finetune_log.csv", index=False)



        if val_loss < best_val:

            best_val = val_loss

            torch.save(model.state_dict(), out_dir / "best_model.pt")



        torch.save(model.state_dict(), out_dir / "last_model.pt")



    print("Best val:", best_val)

    print("Saved to:", out_dir)





if __name__ == "__main__":

    main()

