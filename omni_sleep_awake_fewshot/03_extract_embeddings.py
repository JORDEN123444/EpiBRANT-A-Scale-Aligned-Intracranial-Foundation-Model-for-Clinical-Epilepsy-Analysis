
import argparse

import hashlib

from pathlib import Path



import numpy as np

import pandas as pd

import torch

from torch.utils.data import DataLoader, Dataset



# These two functions are used only because they reconstruct the exact

# architecture compatible with final_pretrained.pt.

# No SOZ labels, predictions, or LOPO checkpoints are used.

from downstream.clinical_soz_localization.model_soz import (

    build_soz_model,

    load_checkpoint_shape_safe,

)





class SegmentDataset(Dataset):

    """

    Loads one preprocessed 60-second SEEG segment.



    Expected cached signal:

        x: [128, 60 * 250]



    Returned model input:

        x: [128, 60, 250]

    """



    def __init__(self, manifest_csv):

        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)



        if self.df.empty:

            raise RuntimeError("The segment manifest is empty.")



    def __len__(self):

        return len(self.df)



    def __getitem__(self, index):

        row = self.df.iloc[index]



        cache_path = None



        for path_column in (

            "cache_path",

            "segment_path",

            "npz_path",

            "preprocessed_path",

        ):

            if (

                path_column in row.index

                and not pd.isna(row[path_column])

            ):

                candidate_path = str(

                    row[path_column]

                ).strip()



                if candidate_path:

                    cache_path = candidate_path

                    break



        if cache_path is None:

            raise KeyError(

                "No valid preprocessed-segment path was found. "

                "Expected one of: cache_path, segment_path, "

                "npz_path, or preprocessed_path."

            )

        cached = np.load(cache_path, allow_pickle=True)



        x = cached["x"].astype(np.float32)

        channel_mask = cached["channel_mask"].astype(np.float32)



        context_sec = int(round(float(row["context_sec"])))

        target_fs = int(round(float(row["target_fs"])))



        expected_samples = context_sec * target_fs



        if x.shape[-1] > expected_samples:

            x = x[:, :expected_samples]



        elif x.shape[-1] < expected_samples:

            missing = expected_samples - x.shape[-1]

            x = np.pad(

                x,

                ((0, 0), (0, missing)),

                mode="constant",

            )



        # [channels, seconds, samples per one-second patch]

        x = x.reshape(

            x.shape[0],

            context_sec,

            target_fs,

        )



        return {

            "x": torch.from_numpy(x),

            "channel_mask": torch.from_numpy(channel_mask),

            "index": torch.tensor(index, dtype=torch.long),

        }





def build_foundation_model(checkpoint_path, device):

    """

    Reconstructs the architecture expected by final_pretrained.pt.



    The wrapper may contain a randomly initialized downstream head, but that

    head is never used. Features are captured directly from the pretrained

    channel encoder.

    """



    cfg = {

        "data": {

            "context_sec": 60,

            "patch_sec": 1,

            "target_fs": 250,

            "max_channels": 128,

        },

        "model": {

            "d_model": 192,

            "seq_len": 60,

            "max_channels": 128,

            "temporal_layers": 3,

            "channel_layers": 2,

            "n_heads": 6,

            "dim_feedforward": 768,

            "dropout": 0.1,

            "t_align": 8,

            "f_align": 128,

            "f_max": 128.0,

            "conv_channels": 24,

        },

        "train": {

            "pretrained_checkpoint": checkpoint_path,

        },

    }



    model = build_soz_model(cfg)



    model = load_checkpoint_shape_safe(

        model,

        checkpoint_path,

        device=device,

    )



    model.to(device)

    model.eval()



    # The foundation model is frozen during feature extraction.

    for parameter in model.parameters():

        parameter.requires_grad = False



    return model





def find_channel_encoder(model):

    """

    Finds the final pretrained channel encoder module.



    This is the final representation-learning stage before a downstream

    task-specific prediction layer.

    """



    candidates = []



    for name, module in model.named_modules():

        lower = name.lower()



        if "channel_encoder" in lower:

            candidates.append((name, module))



    if not candidates:

        print("\nAvailable model modules:")

        for name, _ in model.named_modules():

            print(" ", name)



        raise RuntimeError(

            "Could not locate the pretrained channel_encoder module."

        )



    # Prefer the highest-level channel_encoder module rather than one of

    # its internal Transformer layers.

    candidates = sorted(

        candidates,

        key=lambda item: (

            item[0].count("."),

            len(item[0]),

        ),

    )



    selected_name, selected_module = candidates[0]



    print("Feature extraction module:", selected_name)



    return selected_module





def collect_candidate_tensors(value, candidates, path="output"):

    """

    Recursively searches nested output structures for tensors.



    The channel encoder may return:

        tensor

        tuple

        list

        dictionary

        nested combinations of these

    """



    if torch.is_tensor(value):

        candidates.append((path, value))

        return



    if isinstance(value, dict):

        for key, nested_value in value.items():

            collect_candidate_tensors(

                nested_value,

                candidates,

                path + "." + str(key),

            )

        return



    if isinstance(value, (tuple, list)):

        for index, nested_value in enumerate(value):

            collect_candidate_tensors(

                nested_value,

                candidates,

                path + "[{}]".format(index),

            )

        return



    # Support simple objects that store outputs as attributes.

    if hasattr(value, "__dict__"):

        for key, nested_value in value.__dict__.items():

            collect_candidate_tensors(

                nested_value,

                candidates,

                path + "." + str(key),

            )





def select_latent_tensor(channel_encoder_output, d_model=192):

    """

    Selects the latent feature tensor whose final dimension equals d_model.

    """



    candidates = []

    collect_candidate_tensors(

        channel_encoder_output,

        candidates,

    )



    valid = []



    for path, tensor in candidates:

        if tensor.dim() >= 2 and tensor.shape[-1] == d_model:

            valid.append((path, tensor))



    if not valid:

        print("\nTensors found in channel encoder output:")



        if not candidates:

            print("  No tensors were found.")

        else:

            for path, tensor in candidates:

                print(" ", path, tuple(tensor.shape))



        raise RuntimeError(

            "No latent channel-encoder tensor with final dimension {} "

            "was found.".format(d_model)

        )



    # Prefer the largest matching tensor, usually [B,C,T,D].

    valid = sorted(

        valid,

        key=lambda item: item[1].numel(),

        reverse=True,

    )



    selected_path, selected_tensor = valid[0]



    print_once = not hasattr(

        select_latent_tensor,

        "_printed",

    )



    if print_once:

        print(

            "Selected latent tensor:",

            selected_path,

            tuple(selected_tensor.shape),

        )

        select_latent_tensor._printed = True



    return selected_tensor





def standardize_feature_shape(

    features,

    batch_size,

    channels,

    time_steps,

    d_model,

):

    """

    Convert the channel-encoder output into a standard representation.



    Supported layouts include:



        [B, C, T, D]

        [B, T, C, D]

        [C, B*T, D]

        [B*T, C, D]

        [B*C, T, D]

        [B, C, D]

        [B, tokens, D]

        [B, D]



    The required output for channel-aware temporal pooling is preferably:



        [B, C, T, D]

    """



    shape = tuple(features.shape)



    if features.shape[-1] != d_model:

        raise RuntimeError(

            "Feature dimension does not match d_model={}: {}"

            .format(d_model, shape)

        )



    # ------------------------------------------------------------

    # Four-dimensional layouts

    # ------------------------------------------------------------

    if features.dim() == 4:



        # Already [B, C, T, D]

        if (

            features.shape[0] == batch_size

            and features.shape[1] == channels

            and features.shape[2] == time_steps

        ):

            return features.contiguous()



        # [B, T, C, D] -> [B, C, T, D]

        if (

            features.shape[0] == batch_size

            and features.shape[1] == time_steps

            and features.shape[2] == channels

        ):

            return features.permute(

                0, 2, 1, 3

            ).contiguous()



    # ------------------------------------------------------------

    # Three-dimensional layouts

    # ------------------------------------------------------------

    if features.dim() == 3:



        # The observed channel-encoder output:

        #

        # [C, B*T, D] -> [B, C, T, D]

        #

        # Example:

        # [128, 240, 192], where 240 = 4 * 60

        if (

            features.shape[0] == channels

            and features.shape[1] == batch_size * time_steps

        ):

            features = features.permute(

                1, 0, 2

            ).contiguous()



            # [B*T, C, D] -> [B, T, C, D]

            features = features.reshape(

                batch_size,

                time_steps,

                channels,

                d_model,

            )



            # [B, T, C, D] -> [B, C, T, D]

            return features.permute(

                0, 2, 1, 3

            ).contiguous()



        # [B*T, C, D] -> [B, C, T, D]

        if (

            features.shape[0] == batch_size * time_steps

            and features.shape[1] == channels

        ):

            features = features.reshape(

                batch_size,

                time_steps,

                channels,

                d_model,

            )



            return features.permute(

                0, 2, 1, 3

            ).contiguous()



        # [B*C, T, D] -> [B, C, T, D]

        if (

            features.shape[0] == batch_size * channels

            and features.shape[1] == time_steps

        ):

            return features.reshape(

                batch_size,

                channels,

                time_steps,

                d_model,

            ).contiguous()



        # [B, C, D] or [B, tokens, D]

        if features.shape[0] == batch_size:

            return features.contiguous()



    # ------------------------------------------------------------

    # Two-dimensional layouts

    # ------------------------------------------------------------

    if features.dim() == 2:



        # Already one embedding per segment: [B, D]

        if features.shape[0] == batch_size:

            return features.contiguous()



        # [B*C*T, D] -> [B, C, T, D]

        if (

            features.shape[0]

            == batch_size * channels * time_steps

        ):

            return features.reshape(

                batch_size,

                channels,

                time_steps,

                d_model,

            ).contiguous()



    raise RuntimeError(

        "Could not standardize channel-encoder output shape: {}. "

        "Expected batch={}, channels={}, time_steps={}, d_model={}"

        .format(

            shape,

            batch_size,

            channels,

            time_steps,

            d_model,

        )

    )





def pool_foundation_features(features, channel_mask, d_model=192):

    """

    Pools the latent foundation representation into one 192-dimensional

    embedding per 60-second segment.



    Padded channels are excluded using channel_mask.

    """



    if features.dim() == 4:

        # [B,C,T,D]

        if features.shape[1] == channel_mask.shape[1]:

            weights = channel_mask[

                :,

                :,

                None,

                None,

            ].to(features.dtype)



            weights = weights.expand(

                features.shape[0],

                features.shape[1],

                features.shape[2],

                1,

            )



            numerator = (

                features * weights

            ).sum(dim=(1, 2))



            denominator = weights.sum(

                dim=(1, 2)

            ).clamp_min(1.0)



            return numerator / denominator



        # Fallback if the channel dimension cannot be identified.

        return features.mean(dim=(1, 2))



    if features.dim() == 3:

        # [B,C,D]

        if features.shape[1] == channel_mask.shape[1]:

            weights = channel_mask[

                :,

                :,

                None,

            ].to(features.dtype)



            numerator = (

                features * weights

            ).sum(dim=1)



            denominator = weights.sum(

                dim=1

            ).clamp_min(1.0)



            return numerator / denominator



        # [B,tokens,D]

        return features.mean(dim=1)



    if features.dim() == 2:

        # [B,D]

        return features



    raise RuntimeError(

        "Unexpected standardized feature shape: {}"

        .format(tuple(features.shape))

    )





@torch.no_grad()

def main():

    parser = argparse.ArgumentParser()



    parser.add_argument(

        "--manifest",

        required=True,

    )



    parser.add_argument(

        "--checkpoint",

        required=True,

    )



    parser.add_argument(

        "--out_dir",

        required=True,

    )



    parser.add_argument(

        "--batch_size",

        type=int,

        default=4,

    )



    args = parser.parse_args()



    out_dir = Path(args.out_dir)

    embedding_dir = out_dir / "foundation_embeddings"



    embedding_dir.mkdir(

        parents=True,

        exist_ok=True,

    )



    dataset = SegmentDataset(args.manifest)



    loader = DataLoader(

        dataset,

        batch_size=args.batch_size,

        shuffle=False,

        num_workers=0,

        pin_memory=True,

    )



    device = torch.device(

        "cuda" if torch.cuda.is_available() else "cpu"

    )



    print("Device:", device)

    print("Loading foundation checkpoint:", args.checkpoint)

    print("Number of segments:", len(dataset))



    foundation_model = build_foundation_model(

        args.checkpoint,

        device,

    )



    channel_encoder = find_channel_encoder(

        foundation_model

    )



    captured = {}



    def channel_encoder_hook(module, inputs, output):

        captured["output"] = output



    hook_handle = channel_encoder.register_forward_hook(

        channel_encoder_hook

    )



    embedding_paths = [""] * len(dataset)



    try:

        for batch_number, batch in enumerate(

            loader,

            start=1,

        ):

            x = batch["x"].to(

                device,

                non_blocking=True,

            )



            channel_mask = batch["channel_mask"].to(

                device,

                non_blocking=True,

            )



            batch_size = x.shape[0]

            number_of_channels = x.shape[1]

            number_of_patches = x.shape[2]



            sampling_rate = torch.full(

                (batch_size,),

                250.0,

                dtype=x.dtype,

                device=device,

            )



            captured.clear()



            # Run the checkpoint-compatible forward path.

            # Its downstream output is discarded.

            _ = foundation_model(

                x,

                sampling_rate,

                channel_mask,

            )



            if "output" not in captured:

                raise RuntimeError(

                    "The channel encoder hook did not capture an output."

                )



            latent = select_latent_tensor(

                captured["output"],

                d_model=192,

            )



            latent = standardize_feature_shape(

                latent,

                batch_size=batch_size,

                channels=number_of_channels,

                time_steps=number_of_patches,

                d_model=192,

            )



            pooled = pool_foundation_features(

                latent,

                channel_mask,

                d_model=192,

            )



            if pooled.shape != (batch_size, 192):

                raise RuntimeError(

                    "Expected pooled embeddings [B,192], received {}"

                    .format(tuple(pooled.shape))

                )



            pooled = pooled.detach().cpu().numpy().astype(

                np.float32

            )



            dataframe_indices = batch["index"].numpy()



            for local_index, dataframe_index in enumerate(

                dataframe_indices

            ):

                row = dataset.df.iloc[int(dataframe_index)]



                key = "{}|{}|{}".format(

                    row["cache_path"],

                    row["patient_id"],

                    row["segment_id"],

                )



                file_hash = hashlib.md5(

                    key.encode("utf-8")

                ).hexdigest()



                embedding_path = (

                    embedding_dir

                    / "{}.npy".format(file_hash)

                )



                np.save(

                    embedding_path,

                    pooled[local_index],

                )



                embedding_paths[int(dataframe_index)] = str(

                    embedding_path

                )



            if (

                batch_number == 1

                or batch_number % 20 == 0

                or batch_number == len(loader)

            ):

                print(

                    "Processed batch {}/{} | embedding shape {}"

                    .format(

                        batch_number,

                        len(loader),

                        tuple(pooled.shape),

                    )

                )



    finally:

        hook_handle.remove()



    output_manifest = dataset.df.copy()

    output_manifest["embedding_path"] = embedding_paths



    missing_embeddings = (

        output_manifest["embedding_path"] == ""

    ).sum()



    if missing_embeddings:

        raise RuntimeError(

            "{} embeddings were not created."

            .format(missing_embeddings)

        )



    output_path = out_dir / "embedding_manifest.csv"



    output_manifest.to_csv(

        output_path,

        index=False,

    )



    print("\nFeature extraction completed.")

    print("Embedding dimension: 192")

    print("Total embeddings:", len(output_manifest))

    print("Saved manifest:", output_path)





if __name__ == "__main__":

    main()

