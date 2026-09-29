import torch
import torch.nn as nn


from models.epibrant500m_pretrain import EpiBRANT500M



class EpiBRANTSOZLocalizer(nn.Module):
    """
    EpiBRANT500M based clinical SOZ localization model.

    Input:
        signal:
            [B, 288, T]

        channel_mask:
            [B, 288]

        fs:
            sampling frequency


    Output:

        contact_logits:
            [B,288]

        One probability for each contact
    """



    def __init__(
        self,
        checkpoint_path,
        dropout=0.2,
        freeze_encoder=True,
    ):

        super().__init__()



        print("="*80)
        print("LOADING EpiBRANT500M FOR SOZ LOCALIZATION")
        print("="*80)



        # -------------------------------------------------
        # Load pretrained architecture
        # -------------------------------------------------

        pretrained = EpiBRANT500M()



        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )


        if "model" not in checkpoint:

            raise RuntimeError(
                "Checkpoint missing model key"
            )



        result = pretrained.load_state_dict(
            checkpoint["model"],
            strict=True
        )


        print(
            "Missing keys:",
            result.missing_keys
        )


        print(
            "Unexpected keys:",
            result.unexpected_keys
        )


        print(
            "Checkpoint loading SUCCESS"
        )



        # -------------------------------------------------
        # Keep EpiBRANT backbone
        # -------------------------------------------------

        self.scale_alignment = (
            pretrained.scale_alignment
        )


        self.temporal_encoder = (
            pretrained.temporal_encoder
        )


        self.channel_encoder = (
            pretrained.channel_encoder
        )


        self.d_model = 1024



        # -------------------------------------------------
        # SOZ localization head
        #
        # Input:
        # each contact embedding
        #
        # [B,288,1024]
        #
        # Output:
        #
        # [B,288]
        #
        # -------------------------------------------------

        self.soz_head = nn.Sequential(

            nn.LayerNorm(
                self.d_model
            ),


            nn.Dropout(
                dropout
            ),


            nn.Linear(
                self.d_model,
                1
            )

        )



        if freeze_encoder:

            self.freeze_backbone()



    # =====================================================
    # Freeze encoder
    # =====================================================


    def freeze_backbone(self):

        for module in [

            self.scale_alignment,

            self.temporal_encoder,

            self.channel_encoder

        ]:


            for p in module.parameters():

                p.requires_grad = False



    # =====================================================
    # Unfreeze encoder
    # =====================================================


    def unfreeze_backbone(self):

        for module in [

            self.scale_alignment,

            self.temporal_encoder,

            self.channel_encoder

        ]:


            for p in module.parameters():

                p.requires_grad = True



    # =====================================================
    # Contact-level embedding extraction
    # =====================================================


    def encode_contacts(
        self,
        signal,
        fs,
        channel_mask
    ):

        """
        Return contact-level embeddings.

        Input:

            signal:
                [B,288,T]


        Output:

            embeddings:
                [B,288,1024]

        """



        # ---------------------------------------------
        # Scale alignment
        # ---------------------------------------------


        tokens = self.scale_alignment(
            signal,
            fs
        )


        # expected:

        # [B,C,time_tokens,1024]



        # ---------------------------------------------
        # Apply channel mask
        # ---------------------------------------------


        mask = (

            channel_mask

            .to(
                device=tokens.device,
                dtype=tokens.dtype
            )

            .unsqueeze(-1)

            .unsqueeze(-1)

        )



        tokens = tokens * mask



        # ---------------------------------------------
        # Temporal transformer
        # ---------------------------------------------


        latent = self.temporal_encoder(
            tokens
        )



        # ---------------------------------------------
        # Channel transformer
        # ---------------------------------------------


        latent = self.channel_encoder(
            latent,
            channel_mask=channel_mask
        )


        # latent:

        # [B,288,time_tokens,1024]



        # ---------------------------------------------
        # Average only temporal dimension
        # keep channel dimension
        # ---------------------------------------------


        contact_embeddings = latent.mean(
            dim=2
        )


        # output:

        # [B,288,1024]


        return contact_embeddings



    # =====================================================
    # Forward
    # =====================================================


    def forward(
        self,
        signal,
        fs,
        channel_mask
    ):


        embeddings = self.encode_contacts(

            signal,

            fs,

            channel_mask

        )


        logits = self.soz_head(
            embeddings
        )


        # remove last dimension

        logits = logits.squeeze(-1)


        # [B,288]


        return logits