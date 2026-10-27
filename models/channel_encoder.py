import torch
import torch.nn as nn


class ChannelEncoder(nn.Module):

    """
    Channel-count-independent SEEG Transformer.

    Input:
        x:
            [B,C,K,D]

        channel_mask:
            [B,C]
            True  = real SEEG contact
            False = padded channel

    Output:
        [B,C,K,D]
    """

    def __init__(
        self,
        d_model=1024,
        depth=8,
        heads=16,
        dim_feedforward=4096,
        dropout=0.1,
    ):
        super().__init__()

        self.d_model = d_model

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
        channel_mask=None,
    ):

        B, C, K, D = x.shape

        if D != self.d_model:
            raise ValueError(
                f"Expected D={self.d_model}, got {D}"
            )

        # -----------------------------------------
        # B,C,K,D -> B,K,C,D -> B*K,C,D
        # -----------------------------------------

        x = x.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        x = x.reshape(
            B * K,
            C,
            D,
        )

        padding_mask = None

        if channel_mask is not None:

            if channel_mask.shape != (
                B,
                C,
            ):
                raise ValueError(
                    "channel_mask must have shape "
                    f"[B,C], got {channel_mask.shape}"
                )

            # Transformer uses True = ignore/padding.
            padding_mask = (
                ~channel_mask.bool()
            )

            padding_mask = (
                padding_mask
                .unsqueeze(1)
                .expand(
                    B,
                    K,
                    C,
                )
                .reshape(
                    B * K,
                    C,
                )
            )

        x = self.encoder(
            x,
            src_key_padding_mask=padding_mask,
        )

        x = self.final_norm(x)

        # -----------------------------------------
        # B*K,C,D -> B,K,C,D -> B,C,K,D
        # -----------------------------------------

        x = x.reshape(
            B,
            K,
            C,
            D,
        )

        x = x.permute(
            0,
            2,
            1,
            3,
        ).contiguous()

        return x