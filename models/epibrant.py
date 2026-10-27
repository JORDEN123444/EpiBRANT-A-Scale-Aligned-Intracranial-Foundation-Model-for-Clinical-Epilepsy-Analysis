import torch
import torch.nn as nn


from scale_alignment import ScaleAlignmentEmbedding
from temporal_encoder import TemporalTransformer
from channel_encoder import ChannelTransformer



class MaskedDecoder(nn.Module):

    def __init__(
        self,
        d_model=1024,
        output_dim=1024
    ):

        super().__init__()

        self.decoder = nn.Sequential(

            nn.Linear(
                d_model,
                4096
            ),

            nn.GELU(),

            nn.Linear(
                4096,
                output_dim
            )
        )


    def forward(self,x):

        return self.decoder(x)





class EpiBRANT500M(nn.Module):


    """
    EpiBRANT-500M

    BrainWave-style SEEG foundation model


    Input:

    x:
       B,C,T


    output:

    reconstructed latent representation

    """



    def __init__(

        self,

        d_model=1024,

        temporal_layers=24,

        channel_layers=8,

        heads=16,

        max_channels=288,

        freq_bins=128

    ):

        super().__init__()



        self.d_model=d_model

        self.max_channels=max_channels



        # --------------------------------
        # Scale alignment spectrogram encoder
        # --------------------------------

        self.scale_encoder = ScaleAlignmentEmbedding(

            d_model=d_model,

            freq_bins=freq_bins

        )



        # --------------------------------
        # Temporal transformer
        # --------------------------------

        self.temporal_encoder = TemporalTransformer(

            d_model=d_model,

            layers=temporal_layers,

            heads=heads

        )



        # --------------------------------
        # Channel attention transformer
        # --------------------------------

        self.channel_encoder = ChannelTransformer(

            d_model=d_model,

            layers=channel_layers,

            heads=heads

        )



        # --------------------------------
        # Masked reconstruction decoder
        # --------------------------------

        self.decoder = MaskedDecoder(

            d_model=d_model,

            output_dim=d_model

        )




    def forward(

        self,

        x,

        sampling_rate,

        channel_mask=None,

        mask=None

    ):


        """
        x:

        B,C,T


        sampling_rate:

        integer


        """



        # -----------------------------
        # Spectrogram scale alignment
        # -----------------------------

        z = self.scale_encoder(

            x,

            sampling_rate

        )


        # z:
        # B,C,60,1024



        # -----------------------------
        # Temporal modeling
        # -----------------------------


        z = self.temporal_encoder(z)



        # -----------------------------
        # Channel modeling
        # -----------------------------


        channel_features = self.channel_encoder(

            z,

            channel_mask

        )



        # channel_features:

        # B,C,1024



        # -----------------------------
        # Reconstruction target
        # -----------------------------


        reconstructed = self.decoder(

            channel_features

        )


        return {


            "latent":

                channel_features,


            "reconstruction":

                reconstructed,


            "temporal_features":

                z

        }