import torch


class SEEGMaskGenerator:

    """
    Random temporal token masking.

    Input:
        B,C,K,D

    Output:
        masked input
        mask index

    """

    def __init__(
        self,
        mask_ratio=0.4
    ):

        self.mask_ratio = mask_ratio



    def __call__(self,x):


        B,C,K,D = x.shape


        mask = torch.rand(
            B,
            C,
            K,
            device=x.device
        ) < self.mask_ratio



        x_masked = x.clone()



        # zero masked tokens

        x_masked[mask] = 0



        return x_masked, mask