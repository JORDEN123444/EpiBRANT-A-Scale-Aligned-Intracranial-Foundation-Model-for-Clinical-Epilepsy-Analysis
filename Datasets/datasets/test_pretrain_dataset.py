from epibrant_pretrain_dataset import EpiBRANTPretrainDataset


dataset = EpiBRANTPretrainDataset(
    "global_pretrain_manifest.csv"
)


sample = dataset[0]


print("===================")
print("SAMPLE TEST")
print("===================")


print(
    "Signal:",
    sample["signal"].shape
)


print(
    "Sampling rate:",
    sample["fs"]
)


print(
    "Subject:",
    sample["subject"]
)