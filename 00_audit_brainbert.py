
import argparse

import inspect

import sys

from pathlib import Path



import numpy as np

import torch

from omegaconf import OmegaConf





def safe_torch_load(path: Path):

    kwargs = {"map_location": "cpu"}



    if "weights_only" in inspect.signature(torch.load).parameters:

        kwargs["weights_only"] = False



    return torch.load(path, **kwargs)





def clean_state_dict(state_dict):

    cleaned = {}



    for key, value in state_dict.items():

        for prefix in ["module.", "model."]:

            if key.startswith(prefix):

                key = key[len(prefix):]



        cleaned[key] = value



    return cleaned





def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--brainbert_root", required=True)

    parser.add_argument("--checkpoint", required=True)

    args = parser.parse_args()



    root = Path(args.brainbert_root)

    checkpoint_path = Path(args.checkpoint)



    if not root.exists():

        raise FileNotFoundError(root)



    if not checkpoint_path.exists():

        raise FileNotFoundError(checkpoint_path)



    sys.path.insert(0, str(root))



    from models import build_model

    from preprocessors import build_preprocessor



    checkpoint = safe_torch_load(checkpoint_path)



    print("=" * 100)

    print("BRAINBERT CHECKPOINT AUDIT")

    print("=" * 100)

    print("Repository:", root)

    print("Checkpoint:", checkpoint_path)

    print("Checkpoint bytes:", checkpoint_path.stat().st_size)

    print("Top-level type:", type(checkpoint).__name__)



    if isinstance(checkpoint, dict):

        print("Top-level keys:", list(checkpoint.keys()))



    if isinstance(checkpoint, dict) and "model_cfg" in checkpoint:

        model_cfg = checkpoint["model_cfg"]

        print("Using model_cfg from checkpoint.")

    else:

        model_cfg = OmegaConf.load(

            root / "conf/model/masked_tf_model_large.yaml"

        )

        print("Using official masked_tf_model_large.yaml.")



    print("\nModel configuration:")

    print(model_cfg)



    model = build_model(model_cfg)



    if isinstance(checkpoint, dict) and "model" in checkpoint:

        state_dict = checkpoint["model"]

    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:

        state_dict = checkpoint["state_dict"]

    elif isinstance(checkpoint, dict):

        state_dict = checkpoint

    else:

        raise RuntimeError("No state dictionary was identified.")



    state_dict = clean_state_dict(state_dict)



    model_state = model.state_dict()



    compatible = {

        key: value

        for key, value in state_dict.items()

        if (

            key in model_state

            and torch.is_tensor(value)

            and tuple(value.shape) == tuple(model_state[key].shape)

        )

    }



    missing, unexpected = model.load_state_dict(

        compatible,

        strict=False,

    )



    coverage = len(compatible) / max(1, len(model_state))



    print("\nModel tensors:", len(model_state))

    print("Compatible checkpoint tensors:", len(compatible))

    print("Tensor coverage:", round(coverage, 4))

    print("Missing:", len(missing))

    print("Unexpected:", len(unexpected))



    if coverage < 0.80:

        raise RuntimeError(

            "Checkpoint coverage is below 80%. "

            "Verify that this is the pretrained BrainBERT checkpoint."

        )



    preprocessor_cfg = OmegaConf.load(

        root / "conf/preprocessor/stft.yaml"

    )



    preprocessor = build_preprocessor(

        preprocessor_cfg

    )



    # Official input example duration: 5 seconds at 2048 Hz.

    waveform = np.random.randn(

        5 * 2048

    ).astype(np.float32)



    spectrogram = preprocessor(

        waveform

    )



    if spectrogram.ndim != 2:

        raise RuntimeError(

            f"Expected [sequence, frequency], found {spectrogram.shape}"

        )



    model.eval()



    with torch.no_grad():

        hidden = model(

            spectrogram.unsqueeze(0),

            src_key_mask=None,

            intermediate_rep=True,

        )



    print("\nWaveform shape:", waveform.shape)

    print("Spectrogram shape:", tuple(spectrogram.shape))

    print("Hidden representation:", tuple(hidden.shape))

    print("Expected hidden dimension:", model_cfg.hidden_dim)



    assert hidden.ndim == 3

    assert hidden.shape[-1] == int(model_cfg.hidden_dim)

    assert torch.isfinite(hidden).all()



    print("\nBRAINBERT AUDIT PASSED")





if __name__ == "__main__":

    main()

