import torch
import torch.nn as nn

from .scale_alignment import ScaleAlignmentEmbedding
from .temporal_encoder import TemporalEncoder
from .channel_encoder import ChannelEncoder
from .masked_decoder import MaskedDecoder


class EpiBRANT500M(nn.Module):
    """
    EpiBRANT-500M self-supervised pretraining model.

    Default architecture
    --------------------
    d_model            = 1024
    temporal layers    = 20
    channel layers     = 12
    decoder layers     = 7
    attention heads    = 16
    mask ratio         = 40%

    Expected parameters:
        ~500.8 million
    """

    def __init__(
        self,
        d_model=1024,
        freq_bins=128,
        time_bins=8,
        temporal_depth=20,
        channel_depth=12,
        decoder_depth=7,
        heads=16,
        mask_ratio=0.40,
        f_max=128.0,
    ):
        super().__init__()

        self.d_model = d_model
        self.mask_ratio = mask_ratio

        # =====================================================
        # SCALE-ALIGNED SPECTROTEMPORAL EMBEDDING
        # =====================================================

        self.scale_alignment = ScaleAlignmentEmbedding(
            d_model=d_model,
            freq_bins=freq_bins,
            time_bins=time_bins,
            f_max=f_max,
        )

        # =====================================================
        # LEARNABLE MASK TOKEN
        # =====================================================

        self.mask_token = nn.Parameter(
            torch.zeros(
                1,
                1,
                1,
                d_model,
            )
        )

        nn.init.trunc_normal_(
            self.mask_token,
            std=0.02,
        )

        # =====================================================
        # TEMPORAL TRANSFORMER
        # 20 layers
        # =====================================================

        self.temporal_encoder = TemporalEncoder(
            d_model=d_model,
            depth=temporal_depth,
            heads=heads,
            dim_feedforward=4 * d_model,
            max_patches=60,
        )

        # =====================================================
        # CHANNEL TRANSFORMER
        # 12 layers
        # =====================================================

        self.channel_encoder = ChannelEncoder(
            d_model=d_model,
            depth=channel_depth,
            heads=heads,
            dim_feedforward=4 * d_model,
        )

        # =====================================================
        # MASKED RECONSTRUCTION DECODER
        # 7 layers
        # =====================================================

        self.decoder = MaskedDecoder(
            d_model=d_model,
            depth=decoder_depth,
            heads=heads,
        )

    # =========================================================
    # MASK GENERATION
    # =========================================================

    def generate_mask(
        self,
        tokens,
        channel_mask=None,
    ):
        """
        Generate token-level masking.

        tokens:
            [B,C,K,D]

        channel_mask:
            [B,C]
            True  = real SEEG electrode
            False = padded electrode

        returns:
            [B,C,K]
        """

        B, C, K, _ = tokens.shape

        mask = (
            torch.rand(
                B,
                C,
                K,
                device=tokens.device,
            )
            < self.mask_ratio
        )

        # Never mask padded channels because they should not
        # participate in the reconstruction objective.
        if channel_mask is not None:

            mask = (
                mask
                & channel_mask.bool().unsqueeze(-1)
            )

        return mask

    # =========================================================
    # FORWARD
    # =========================================================

    def forward(
        self,
        x,
        fs,
        channel_mask=None,
    ):
        """
        x:
            [B,C,T]

        fs:
            native sampling frequency

        channel_mask:
            [B,C]
        """

        # =====================================================
        # 1. SCALE ALIGNMENT
        # =====================================================

        tokens = self.scale_alignment(
            x,
            fs,
        )

        B, C, K, D = tokens.shape

        # =====================================================
        # 2. VALID CHANNEL MASK
        # =====================================================

        if channel_mask is None:

            channel_mask = torch.ones(
                B,
                C,
                dtype=torch.bool,
                device=tokens.device,
            )

        else:

            channel_mask = channel_mask.bool()

        if channel_mask.shape != (B, C):

            raise ValueError(
                f"channel_mask should have shape {(B,C)}, "
                f"but got {tuple(channel_mask.shape)}"
            )

        # Zero representations belonging to padded contacts.
        valid = (
            channel_mask
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

        tokens = tokens * valid

        # =====================================================
        # 3. RECONSTRUCTION TARGET
        # =====================================================

        target = tokens.detach()

        # =====================================================
        # 4. GENERATE MASK BEFORE ENCODERS
        # =====================================================

        mask = self.generate_mask(
            tokens,
            channel_mask,
        )

        # =====================================================
        # 5. INSERT LEARNABLE MASK TOKEN
        # =====================================================

        mask_token = self.mask_token.expand(
            B,
            C,
            K,
            D,
        )

        masked_tokens = torch.where(
            mask.unsqueeze(-1),
            mask_token,
            tokens,
        )

        # Keep padded channels zero.
        masked_tokens = (
            masked_tokens
            * valid
        )

        # =====================================================
        # 6. TEMPORAL TRANSFORMER
        # =====================================================

        z = self.temporal_encoder(
            masked_tokens
        )

        # Keep padded contacts zero.
        z = z * valid

        # =====================================================
        # 7. CHANNEL TRANSFORMER
        # =====================================================

        z = self.channel_encoder(
            z,
            channel_mask=channel_mask,
        )

        # Again ensure padded channels cannot acquire features.
        z = z * valid

        # =====================================================
        # 8. RECONSTRUCTION
        # =====================================================

        reconstruction = self.decoder(
            z
        )

        reconstruction = (
            reconstruction
            * valid
        )

        return {

            "tokens":
                tokens,

            "target":
                target,

            "masked_tokens":
                masked_tokens,

            "latent":
                z,

            "reconstruction":
                reconstruction,

            "mask":
                mask,

            "channel_mask":
                channel_mask,
        }