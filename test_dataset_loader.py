from epibrant_pretrain_dataset import EpiBRANTPretrainDataset



dataset=EpiBRANTPretrainDataset(

    "global_pretrain_manifest.csv"

)



sample=dataset[0]


print(sample["signal"].shape)

print(sample["fs"])

print(sample["subject"])