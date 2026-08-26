import torch
import torch.nn as nn



class TemporalTransformer(nn.Module):


    def __init__(
        self,
        d_model=1024,
        layers=24,
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



    def forward(self,x):


        B,C,K,D=x.shape



        x=x.reshape(
            B*C,
            K,
            D
        )


        x=self.encoder(x)



        x=x.reshape(
            B,
            C,
            K,
            D
        )


        return x