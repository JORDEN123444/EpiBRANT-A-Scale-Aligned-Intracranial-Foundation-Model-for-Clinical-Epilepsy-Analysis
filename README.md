Below is a professional GitHub `README.md` for your **EpiBRANT / SEEG Foundation Model pretraining pipeline**. I kept it suitable for a research repository and **did not mention patient/subject numbers**, as requested.

```markdown
# EpiBRANT: Intracranial EEG Foundation Model Pretraining Framework

Official implementation of the self-supervised pretraining pipeline for **EpiBRANT**, a large-scale intracranial EEG (iEEG/SEEG) foundation model designed to learn generalizable neural representations from heterogeneous intracranial recordings.

The framework follows a BrainWave-inspired foundation model paradigm and enables representation learning from large-scale unlabeled SEEG recordings through masked spectrogram modeling, scale alignment, temporal modeling, and adaptive channel representation learning.

---

# Overview

Intracranial EEG recordings provide rich information about epileptic brain dynamics but present significant challenges due to:

- heterogeneous acquisition systems,
- different sampling frequencies,
- variable electrode configurations,
- inconsistent recording durations,
- limited expert annotations.

EpiBRANT addresses these challenges through a self-supervised pretraining framework that learns intrinsic neural representations directly from raw SEEG signals without requiring seizure labels or manual annotations.

The pretraining pipeline consists of:

```

Raw SEEG recordings
|
↓
Dataset indexing and global manifest construction
|
↓
Native sampling frequency preservation
|
↓
60-second signal window extraction
|
↓
Time-frequency representation generation
|
↓
Scale alignment embedding
|
↓
Temporal Transformer encoder
|
↓
Adaptive channel attention encoder
|
↓
Masked spectrogram reconstruction
|
↓
Pretrained SEEG foundation encoder

```

---

# Repository Structure

```



The analysis preserves original acquisition characteristics and avoids unnecessary resampling.

---

# 2. Global Manifest Generation

A unified manifest is created to organize heterogeneous SEEG recordings.

The manifest contains:

* dataset source,
* subject identifier,
* session identifier,
* recording path,
* sampling frequency,
* recording duration,
* channel information.

Run:

```bash
python scripts/create_global_manifest.py
```

Output:

```
global_pretrain_manifest.csv
```

The manifest enables scalable dataset loading while maintaining recording-level metadata.

---

# 3. SEEG Signal Loading

The pretraining dataloader supports:

* EDF-based intracranial recordings,
* different sampling frequencies,
* variable electrode numbers,
* dynamic channel handling.

Example test:

```bash
python datasets/test_dataset_loader.py
```

Expected output:

```
Signal shape:
[channel, samples]

Sampling frequency:
native recording frequency
```

---

# 4. Scale Alignment Representation

Different SEEG systems use different sampling rates.

Instead of forcing all recordings into a fixed sampling frequency, EpiBRANT applies a scale alignment module.

The module:

1. divides continuous recordings into fixed temporal patches,
2. generates spectrogram representations,
3. aligns frequency dimensions,
4. projects representations into a unified latent space.

Input:

```
B × C × T
```

Output:

```
B × C × temporal patches × embedding dimension
```

This allows heterogeneous SEEG recordings to share a common representation space.

---

# 5. Spectrogram Representation

Each temporal segment is transformed into a time-frequency representation.

Processing steps:

```
Raw SEEG signal

      ↓

STFT

      ↓

Power spectrum

      ↓

Log transformation

      ↓

Frequency alignment

      ↓

Spectrogram embedding
```

A Hann window is applied during STFT calculation to reduce spectral leakage.

---

# 6. EpiBRANT Backbone Architecture

The foundation model contains:

## Scale Alignment Encoder

Learns frequency-independent representations from heterogeneous sampling rates.

---

## Temporal Transformer Encoder

Models sequential neural dynamics across temporal patches.

Input:

```
Temporal spectrogram tokens
```

Output:

```
Temporal neural representation
```

---

## Channel Attention Encoder

Learns relationships between intracranial contacts.

The design is channel-count adaptive and supports variable electrode configurations.

---

## Masked Reconstruction Decoder

During pretraining:

* portions of spectrogram tokens are masked,
* the encoder predicts hidden representations,
* reconstruction loss guides representation learning.

---

# Pretraining Objective

The model is optimized using self-supervised masked modeling.

The primary objective is:

```
Original spectrogram

        ↓

Random masking

        ↓

Encoder prediction

        ↓

Reconstruction

        ↓

Masked reconstruction loss
```

The objective enables learning without requiring manual clinical annotations.

---

# Model Forward Test

Before large-scale training:

```bash
python models/check_model.py
```

The test verifies:

* architecture initialization,
* tensor dimensions,
* forward propagation,
* reconstruction output.

---

# Training

The pretraining process consists of:

1. Dataset loading
2. Dynamic SEEG sampling
3. Spectrogram generation
4. Mask generation
5. Forward propagation
6. Reconstruction loss calculation
7. Optimization update

Training configuration is controlled through:

```
configs/
```

Example:

```bash
python train.py \
--config configs/epibrant500m.yaml
```

---

# Reproducibility

The repository provides:

* preprocessing scripts,
* dataset preparation scripts,
* model implementation,
* configuration files,
* training utilities,
* validation scripts.

All experiments should be performed using approved datasets and appropriate ethical permissions.


# Requirements

Recommended environment:

* Python >= 3.10
* PyTorch
* MNE-Python
* NumPy
* Pandas
* SciPy
* CUDA-enabled GPU environment

Install dependencies:

bash
pip install -r requirements.txt

Pretrained Model
The pretrained EpiBRANT checkpoint is available at:
Hugging Face
https://huggingface.co/ijaz0310/EpiBRANT

# Citation

If you use this implementation, please cite:


EpiBRANT:
A Self-Supervised Foundation Model for Intracranial EEG Representation Learning


# Acknowledgement

This project builds upon advances in large-scale neural signal representation learning and BrainWave-inspired self-supervised learning frameworks for heterogeneous neural recordings.

This README is suitable for a **public GitHub repository linked in a manuscript submission**, because it explains the complete reproducible pipeline without revealing private cohort details.


