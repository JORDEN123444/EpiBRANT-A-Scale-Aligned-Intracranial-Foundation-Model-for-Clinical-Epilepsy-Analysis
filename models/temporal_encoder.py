import torch
import torch.nn as nn


class TemporalEncoder(nn.Module):

    """
    Temporal Transformer.

    Input:
        [B,C,K,D]

    Output:
        [B,C,K,D]
    """

    def __init__(
        self,
        d_model=1024,
        depth=12,
        heads=16,
        dim_feedforward=4096,
        dropout=0.1,
        max_patches=60,
    ):
        super().__init__()

        self.d_model = d_model
        self.max_patches = max_patches

        self.pos_embedding = nn.Parameter(
            torch.zeros(
                1,
                max_patches,
                d_model,
            )
        )

        nn.init.trunc_normal_(
            self.pos_embedding,
            std=0.02,
        )

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=depth,
        )

        self.final_norm = nn.LayerNorm(
            d_model
        )

    def forward(
        self,
        x,
    ):

        B, C, K, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected D={self.d_model}, got {D}"
            )

        if K > self.max_patches:
            raise ValueError(
                f"K={K} exceeds max_patches={self.max_patches}"
            )

        x = x.reshape(
            B * C,
            K,
            D,
        )

        x = (
            x
            + self.pos_embedding[:, :K]
        )

        x = self.encoder(x)

        x = self.final_norm(x)

        x = x.reshape(
            B,
            C,
            K,
            D,
        )

        return x