import torch
import torch.nn as nn



class ChannelTransformer(nn.Module):


    def __init__(
        self,
        d_model=1024,
        layers=8,
        heads=16
    ):

        super().__init__()



        layer=nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=heads,

            dim_feedforward=4096,

            batch_first=True,

            activation="gelu"

        )



        self.encoder=nn.TransformerEncoder(

            layer,

            num_layers=layers

        )



    def forward(
        self,
        x,
        channel_mask=None
    ):


        B,C,K,D=x.shape


        # average temporal representation

        x=x.mean(2)



        x=self.encoder(x)



        return x