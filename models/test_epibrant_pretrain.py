import torch


from epibrant500m_pretrain import EpiBRANT500M



device="cuda"



model = EpiBRANT500M(

    d_model=1024

).to(device)



model.eval()



signal = torch.randn(

    1,

    256,

    120000

).to(device)



fs = 2000



with torch.no_grad():

    output = model(

        signal,

        fs

    )



print("===================")
print("OUTPUT")
print("===================")



for k,v in output.items():

    print(
        k,
        v.shape
    )