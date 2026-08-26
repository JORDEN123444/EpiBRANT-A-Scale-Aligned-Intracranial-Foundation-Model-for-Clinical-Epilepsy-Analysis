import torch
import torch.nn.functional as F



def masked_reconstruction_loss(
    prediction,
    target,
    mask
):


    loss=F.mse_loss(
        prediction,
        target,
        reduction="none"
    )


    loss=loss.mean(-1)



    loss=(
        loss*mask
    ).sum() / (
        mask.sum()+1e-8
    )



    return loss