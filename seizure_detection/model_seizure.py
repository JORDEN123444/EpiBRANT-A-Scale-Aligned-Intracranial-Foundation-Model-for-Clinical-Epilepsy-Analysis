
import sys

from pathlib import Path



ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT))



import torch

import torch.nn as nn



from model_scale_brant import ScaleAlignedBrantMAE





class SABrantSeizureClassifier(nn.Module):

    def __init__(

        self,

        pretrained_path=None,

        freeze_backbone=False,

        d_model=192,

        seq_len=60,

        max_channels=128,

        temporal_layers=3,

        channel_layers=2,

        n_heads=6,

        dim_feedforward=768,

        dropout=0.1,

        t_align=8,

        f_align=128,

        f_max=128.0,

        conv_channels=24,

    ):

        super().__init__()



        self.backbone = ScaleAlignedBrantMAE(

            d_model=d_model,

            seq_len=seq_len,

            max_channels=max_channels,

            temporal_layers=temporal_layers,

            channel_layers=channel_layers,

            n_heads=n_heads,

            dim_feedforward=dim_feedforward,

            dropout=dropout,

            mask_ratio=0.0,

            t_align=t_align,

            f_align=f_align,

            f_max=f_max,

            conv_channels=conv_channels,

        )



        if pretrained_path is not None:

            state = torch.load(pretrained_path, map_location="cpu")



            # Allow loading a 60-second pretrained checkpoint into shorter-window models,

            # e.g., 5-second seizure detection. We copy the first L positional embeddings.

            model_state = self.backbone.state_dict()



            if "pos_embedding" in state and "pos_embedding" in model_state:

                ckpt_pos = state["pos_embedding"]

                cur_pos = model_state["pos_embedding"]



                if ckpt_pos.shape != cur_pos.shape:

                    new_pos = cur_pos.clone()

                    t = min(ckpt_pos.shape[2], cur_pos.shape[2])

                    new_pos[:, :, :t, :] = ckpt_pos[:, :, :t, :]

                    state["pos_embedding"] = new_pos

                    print(

                        "Adapted pos_embedding:",

                        tuple(ckpt_pos.shape),

                        "->",

                        tuple(new_pos.shape),

                    )



            # Remove any remaining mismatched keys so loading does not crash.

            filtered_state = {}

            skipped = []



            for k, v in state.items():

                if k in model_state and model_state[k].shape == v.shape:

                    filtered_state[k] = v

                else:

                    skipped.append(k)



            missing, unexpected = self.backbone.load_state_dict(filtered_state, strict=False)



            print("Loaded pretrained:", pretrained_path)

            print("Missing:", len(missing))

            print("Unexpected:", len(unexpected))

            print("Skipped mismatched keys:", skipped)



        if freeze_backbone:

            for p in self.backbone.parameters():

                p.requires_grad = False



        self.d_model = d_model



        self.classifier = nn.Sequential(

            nn.LayerNorm(d_model),

            nn.Linear(d_model, d_model),

            nn.GELU(),

            nn.Dropout(0.25),

            nn.Linear(d_model, 1),

        )



    def encode(self, x, fs, channel_mask):

        """

        x: [B, C, L, P]

        fs: [B]

        channel_mask: [B, C]

        """

        b, c, l, p = x.shape



        emb, _ = self.backbone.scale_embed(x, fs)

        emb = emb + self.backbone.pos_embedding[:, :, :l, :]



        # Temporal encoder

        h = emb.reshape(b * c, l, self.d_model).transpose(0, 1)

        h = self.backbone.temporal_encoder(h)

        h = h.transpose(0, 1).reshape(b, c, l, self.d_model)



        # Channel encoder

        h2 = h.permute(0, 2, 1, 3).reshape(b * l, c, self.d_model)

        h2 = h2.transpose(0, 1)



        key_padding = ~channel_mask.bool()

        key_padding = key_padding.unsqueeze(1).expand(b, l, c).reshape(b * l, c)



        z = self.backbone.channel_encoder(h2, src_key_padding_mask=key_padding)

        z = z.transpose(0, 1).reshape(b, l, c, self.d_model).permute(0, 2, 1, 3)



        return z



    def forward(self, x, fs, channel_mask):

        z = self.encode(x, fs, channel_mask)



        # z: [B, C, L, D]

        valid = channel_mask[:, :, None, None]

        z = z * valid



        # mean over channels

        denom = valid.sum(dim=1).clamp(min=1.0)

        z = z.sum(dim=1) / denom



        # mean over time

        z = z.mean(dim=1)



        logit = self.classifier(z).squeeze(-1)

        return logit

