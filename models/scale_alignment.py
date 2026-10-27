import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaleAlignmentEmbedding(nn.Module):
    """
    Sampling-rate-aware scale alignment for heterogeneous SEEG.

    Input
    -----
    x  : [B, C, T]
    fs : native sampling frequency (scalar)

    Output
    ------
    z : [B, C, K, D]

    K = 60 one-second patches
    D = d_model
    """

    def __init__(
        self,
        d_model=1024,
        freq_bins=128,
        time_bins=8,
        patch_seconds=1.0,
        context_seconds=60,
        f_max=128.0,
        stft_chunk=128,
    ):
        super().__init__()

        self.d_model = d_model
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        self.patch_seconds = patch_seconds
        self.context_seconds = context_seconds
        self.f_max = f_max
        self.stft_chunk = stft_chunk

        self.conv_encoder = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),
            nn.GELU(),
        )

        # We average over aligned STFT time after CNN.
        self.projection = nn.Linear(
            64 * freq_bins,
            d_model,
        )

    @staticmethod
    def _scalar_fs(fs):

        if torch.is_tensor(fs):
            if fs.numel() != 1:
                raise ValueError(
                    "ScaleAlignmentEmbedding currently expects "
                    "one sampling frequency per batch."
                )
            fs = fs.detach().cpu().item()

        return float(fs)

    @staticmethod
    def normalize_patches(x):

        mean = x.mean(
            dim=-1,
            keepdim=True,
        )

        std = x.std(
            dim=-1,
            keepdim=True,
            unbiased=False,
        )

        return (
            x - mean
        ) / (
            std + 1e-6
        )

    def forward(
        self,
        x,
        fs,
    ):

        if x.ndim != 3:
            raise ValueError(
                f"Expected [B,C,T], got {tuple(x.shape)}"
            )

        B, C, T = x.shape

        fs = self._scalar_fs(fs)

        patch_size = int(
            round(
                fs * self.patch_seconds
            )
        )

        K = int(
            round(
                self.context_seconds
                / self.patch_seconds
            )
        )

        required_samples = (
            K * patch_size
        )

        if T < required_samples:
            raise ValueError(
                f"Input too short. "
                f"Need {required_samples} samples "
                f"for {self.context_seconds}s at fs={fs}, "
                f"received {T}."
            )

        # Safely trim small rounding excess.
        x = x[..., :required_samples]

        # -----------------------------------------
        # B,C,T -> B,C,K,S
        # -----------------------------------------

        patches = x.reshape(
            B,
            C,
            K,
            patch_size,
        )

        # -----------------------------------------
        # B*C*K,S
        # -----------------------------------------

        patches = patches.reshape(
            B * C * K,
            patch_size,
        )

        patches = self.normalize_patches(
            patches
        )

        # =========================================
        # Sampling-rate-aware STFT
        #
        # physical window ≈ 250 ms
        # physical hop    ≈ 125 ms
        # =========================================

        win_length = max(
            16,
            int(round(patch_size / 4))
        )

        hop_length = max(
            1,
            int(round(patch_size / 8))
        )

        n_fft = win_length

        window = torch.hann_window(
            win_length,
            device=x.device,
            dtype=torch.float32,
        )

        frequency_axis = torch.fft.rfftfreq(
            n_fft,
            d=1.0 / fs,
            device=x.device,
        )

        effective_fmax = min(
            self.f_max,
            fs / 2.0,
        )

        frequency_mask = (
            frequency_axis <= effective_fmax
        )

        if not torch.any(frequency_mask):
            raise RuntimeError(
                f"No STFT bins <= {effective_fmax} Hz."
            )

        encoded_chunks = []

        # =========================================
        # Chunked STFT + CNN
        #
        # Important: do not construct every
        # spectrogram for all contacts at once.
        # =========================================

        for start in range(
            0,
            patches.shape[0],
            self.stft_chunk,
        ):

            end = min(
                start + self.stft_chunk,
                patches.shape[0],
            )

            p = patches[start:end]

            # Keep FFT itself FP32.
            spec = torch.stft(
                p.float(),
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )

            # Power spectrum
            spec = spec.abs().square()

            # Shared physical frequency range
            spec = spec[
                :,
                frequency_mask,
                :
            ]

            # Log-power
            spec = torch.log1p(spec)

            # -------------------------------------
            # Align both frequency and time scales
            # -------------------------------------

            spec = F.interpolate(
                spec.unsqueeze(1),
                size=(
                    self.freq_bins,
                    self.time_bins,
                ),
                mode="bilinear",
                align_corners=False,
            )

            # [N,1,F,T] -> [N,64,F,T]
            z = self.conv_encoder(
                spec
            )

            # Pool aligned STFT time axis only.
            z = z.mean(
                dim=-1
            )

            # [N,64,F] -> [N,64*F]
            z = z.flatten(
                start_dim=1
            )

            z = self.projection(
                z
            )

            encoded_chunks.append(z)

        z = torch.cat(
            encoded_chunks,
            dim=0,
        )

        # -----------------------------------------
        # Restore structure
        # -----------------------------------------

        z = z.reshape(
            B,
            C,
            K,
            self.d_model,
        )

        return z