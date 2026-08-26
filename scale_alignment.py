import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAlignmentEmbedding(nn.Module):

    """
    BrainWave-inspired scale alignment module for heterogeneous SEEG.

    Input:
        x:
            B x C x T

        fs:
            sampling frequency

    Output:
        B x C x 60 x d_model
    """

    def __init__(
        self,
        d_model=1024,
        freq_bins=128,
        n_fft=256,
        hop_length=32,
        patch_seconds=1
    ):

        super().__init__()


        self.d_model = d_model
        self.freq_bins = freq_bins
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.patch_seconds = patch_seconds


        self.conv_encoder = nn.Sequential(

            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1
            ),

            nn.GELU(),


            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1
            ),

            nn.GELU()
        )


        self.projection = nn.Linear(
            64 * freq_bins,
            d_model
        )



    def normalize_signal(self,x):

        mean = x.mean(
            dim=-1,
            keepdim=True
        )

        std = x.std(
            dim=-1,
            keepdim=True
        )


        return (
            x-mean
        ) / (
            std+1e-6
        )



    def compute_spectrogram(self,x):


        window = torch.hann_window(
            self.n_fft,
            device=x.device
        )


        spec = torch.stft(

            x,

            n_fft=self.n_fft,

            hop_length=self.hop_length,

            window=window,

            center=True,

            return_complex=True

        )


        # power spectrum

        spec = torch.abs(spec)**2


        # logarithmic compression

        spec = torch.log1p(spec)


        return spec

    def forward(
            self,
            x,
            fs
    ):
        """
        x:

        B,C,T


        output:

        B,C,60,d_model

        """

        B, C, T = x.shape

        patch_size = int(
            fs * self.patch_seconds
        )

        num_patches = T // patch_size

        if num_patches != 60:
            raise ValueError(
                f"Expected 60 patches but received {num_patches}"
            )

        # --------------------------------------------------
        # Split 60 one-second patches
        # --------------------------------------------------

        patches = x.unfold(
            dimension=-1,
            size=patch_size,
            step=patch_size
        )

        # B,C,60,S

        K = patches.shape[2]

        # --------------------------------------------------
        # Merge batch, channel and temporal patches
        # --------------------------------------------------

        patches = patches.reshape(
            B * C * K,
            patch_size
        )

        # --------------------------------------------------
        # Channel normalization
        # --------------------------------------------------

        patches = self.normalize_signal(
            patches
        )

        # --------------------------------------------------
        # Batched STFT
        # --------------------------------------------------

        window = torch.hann_window(
            self.n_fft,
            device=x.device
        )

        spec = torch.stft(

            patches,

            n_fft=self.n_fft,

            hop_length=self.hop_length,

            window=window,

            center=True,

            return_complex=True

        )

        # B*C*60,F,T

        spec = torch.abs(spec) ** 2

        spec = torch.log1p(
            spec
        )

        # --------------------------------------------------
        # Frequency alignment
        # --------------------------------------------------

        _, freq_dim, time_dim = spec.shape

        spec = F.interpolate(

            spec.unsqueeze(1),

            size=(

                self.freq_bins,

                time_dim

            ),

            mode="bilinear",

            align_corners=False

        )

        # --------------------------------------------------
        # CNN encoder
        # --------------------------------------------------

        z = self.conv_encoder(
            spec
        )

        # remove time dimension

        z = z.mean(
            dim=-1
        )

        z = z.flatten(
            start_dim=1
        )

        z = self.projection(
            z
        )

        # --------------------------------------------------
        # Restore dimensions
        # --------------------------------------------------

        z = z.reshape(

            B,

            C,

            K,

            self.d_model

        )

        return z