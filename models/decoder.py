import torch
import torch.nn as nn


class MaskedSpectrogramDecoder(nn.Module):

    def __init__(
        self,
        d_model=1024,
        output_dim=128
    ):

        super().__init__()

        self.decoder = nn.Sequential(

            nn.Linear(
                d_model,
                2048
            ),

            nn.GELU(),

            nn.Linear(
                2048,
                output_dim
            )
        )


    def forward(self,x):

        return self.decoder(x)