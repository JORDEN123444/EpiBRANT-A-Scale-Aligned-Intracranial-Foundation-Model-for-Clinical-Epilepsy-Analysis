import torch
import torch.nn.functional as F


def masked_reconstruction_loss(
    prediction,
    target,
    mask,
    channel_mask=None,
    loss_type="smooth_l1",
):

    if prediction.shape != target.shape:
        raise ValueError(
            "prediction and target must have identical shapes."
        )

    valid_mask = mask.bool()

    if channel_mask is not None:

        valid_mask = (
            valid_mask
            & channel_mask.bool().unsqueeze(-1)
        )

    if not valid_mask.any():
        raise RuntimeError(
            "No valid masked tokens in this batch."
        )

    # Loss computation in FP32 for numerical stability.
    prediction = prediction.float()
    target = target.float()

    if loss_type == "mse":

        per_element = F.mse_loss(
            prediction,
            target,
            reduction="none",
        )

    elif loss_type == "smooth_l1":

        per_element = F.smooth_l1_loss(
            prediction,
            target,
            reduction="none",
        )

    else:

        raise ValueError(
            f"Unknown loss_type: {loss_type}"
        )

    # Average embedding dimension.
    per_token = per_element.mean(
        dim=-1
    )

    return per_token[
        valid_mask
    ].mean()