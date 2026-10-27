
import argparse

import pandas as pd





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--in_csv", required=True)

    parser.add_argument("--out_csv", required=True)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_per_class", type=int, default=1000)

    args = parser.parse_args()



    df = pd.read_csv(args.in_csv)



    # Keep only rows with usable labels

    df = df[df["label"].isin([0, 1])].copy()



    # Balance total number of normal and seizure windows

    class_counts = df["label"].value_counts()

    n_min = min(class_counts[0], class_counts[1], args.max_per_class)



    balanced = []

    for label in [0, 1]:

        d = df[df["label"] == label].sample(n=n_min, random_state=args.seed)

        balanced.append(d)



    dfb = pd.concat(balanced).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)



    rows = []



    for label in [0, 1]:

        d = dfb[dfb["label"] == label].sample(frac=1.0, random_state=args.seed).reset_index(drop=True)



        n = len(d)

        n_train = int(0.70 * n)

        n_val = int(0.15 * n)



        d.loc[:n_train - 1, "split"] = "train"

        d.loc[n_train:n_train + n_val - 1, "split"] = "val"

        d.loc[n_train + n_val:, "split"] = "test"



        rows.append(d)



    out = pd.concat(rows).sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    out.to_csv(args.out_csv, index=False)



    print("Saved:", args.out_csv)



    print("\nSplit/label counts:")

    print(out.groupby(["split", "label"]).size())



    print("\nFile contribution:")

    print(out.groupby(["split", "original_name", "label"]).size())





if __name__ == "__main__":

    main()

