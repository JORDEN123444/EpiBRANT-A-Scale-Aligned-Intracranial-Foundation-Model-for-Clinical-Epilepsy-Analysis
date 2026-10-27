import torch
import torch.nn as nn



class MaskedDecoder(nn.Module):

    """
    BrainWave-style lightweight decoder.

    Input:

        encoded representation

        B,C,K,D


    Output:

        reconstructed embeddings

        B,C,K,D

    """


    def __init__(
        self,
        d_model=1024,
        depth=4,
        heads=16,
        dropout=0.1
    ):

        super().__init__()



        layer = nn.TransformerEncoderLayer(

            d_model=d_model,

            nhead=heads,

            dim_feedforward=d_model*4,

            dropout=dropout,

            activation="gelu",

            batch_first=True

        )


        self.decoder = nn.TransformerEncoder(

            layer,

            num_layers=depth

        )


        self.output_projection = nn.Linear(

            d_model,

            d_model

        )



    def forward(self,x):

        """

        x:

        B,C,K,D

        """

        B,C,K,D=x.shape


        # flatten contacts

        x=x.reshape(

            B*C,

            K,

            D

        )


        x=self.decoder(x)



        x=self.output_projection(x)



        x=x.reshape(

            B,

            C,

            K,

            D

        )


        return x