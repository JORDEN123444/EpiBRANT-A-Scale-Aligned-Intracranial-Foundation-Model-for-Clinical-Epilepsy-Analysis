import torch

from epibrant500m_pretrain import EpiBRANT500M
from pretrain_loss import masked_reconstruction_loss



device="cuda"



model=EpiBRANT500M().to(device)



signal=torch.randn(

    1,

    256,

    120000

).to(device)



out=model(

    signal,

    2000

)



loss=masked_reconstruction_loss(

    out["reconstruction"],

    out["latent"],

    out["mask"]

)



print("Loss:")

print(loss.item())