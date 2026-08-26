from datasets.epibrant_pretrain_dataset import EpiBRANTPretrainDataset



dataset=EpiBRANTPretrainDataset(

    "/home/ubuntu/Ijaz/Revised code /BrainWae model /SEEG_EpiBRANT500M/global_pretrain_manifest.csv"

)



sample=dataset[0]


print(sample["signal"].shape)

print(sample["fs"])

print(sample["subject"])