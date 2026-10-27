import torch

from masked_decoder import MaskedDecoder



device="cuda"



x=torch.randn(

    1,

    256,

    60,

    1024

).to(device)



model=MaskedDecoder(

    d_model=1024,

    depth=4,

    heads=16

).to(device)



model.eval()



with torch.no_grad():

    y=model(x)



print("Input:")
print(x.shape)


print("Output:")
print(y.shape)