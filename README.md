# LoRa RFFI: Residual CNN and Transformer Comparison

This repository contains a reproducible closed-set LoRa radio-frequency
fingerprint identification (RFFI) study. It reproduces the Residual CNN
baseline developed by Shen, Zhang, and co-authors and compares it with four
progressively refined Transformer models for channel-independent LoRa
spectrograms.

The implementation extends the original
[gxhen/LoRa_RFFI](https://github.com/gxhen/LoRa_RFFI) codebase with:

- Transformer V1: coarse non-overlapping spectrogram patches;
- Transformer V2: finer patches and increased embedding capacity;
- Transformer V3: boundary-preserving padding and a convolutional stem;
- Transformer V4: multi-scale residual context followed by global attention;
- automated training and test logging;
- classification reports, confusion matrices, convergence histories,
  parameter counts, FLOP estimates, training time, and inference latency;
- multi-seed evaluation of the Residual CNN and Transformer V4.

## Main closed-set results

| Model | Test accuracy | Trainable parameters |
| --- | ---: | ---: |
| Residual CNN | 99.42% +/- 0.14% | 12,475,422 |
| Transformer V1 | 84.62% | 192,222 |
| Transformer V2 | 92.88% | 402,014 |
| Transformer V3 | 96.97% | 453,950 |
| Transformer V4 | 99.05% +/- 0.09% | 464,254 |

The CNN and V4 values are the mean and sample standard deviation across random
seeds 2024, 2025, and 2026. The remaining Transformer values describe the
archived single-run progressive comparison.

## Repository structure

```text
Closed_set_RFFI/
  main.py                    Training and test entry point
  dataset_preparation.py     IQ loading and spectrogram preprocessing
  deep_learning_models.py    Residual CNN baseline
  transformer*_models.py     Transformer V1-V4 definitions
  compare_models.py          Common evaluation and complexity analysis
  evaluate_*.py              Standalone evaluation scripts
  experiments/               Metrics, histories and result figures
  comparison_figures/        Cross-model comparison figures and tables
Openset_RFFI_TIFS/            Original open-set reference implementation
```

## Environment

The experiments were run on Windows with Python 3.7, TensorFlow 2.10 and an
NVIDIA RTX 3060 Laptop GPU. Install the core Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

Native Windows GPU support for TensorFlow 2.10 requires a compatible NVIDIA
driver, CUDA 11.2 and cuDNN 8.1.

## Dataset

The IQ dataset is not included in this repository. Obtain the LoRa RFFI data
from the original project and arrange it as:

```text
dataset/
  Train/dataset_training_aug.h5
  Test/dataset_seen_devices.h5
```

Place the dataset under `Closed_set_RFFI/dataset/`, or set the
`LORA_RFFI_DATASET_DIR` environment variable to the directory containing the
`Train` and `Test` folders.

## Running an experiment

Select the requested operation and model near the bottom of
`Closed_set_RFFI/main.py`, then run:

```bash
python Closed_set_RFFI/main.py
```

Generated metrics and figures are archived under
`Closed_set_RFFI/experiments/`. Model weights and datasets are intentionally
excluded from version control because of their size.

## Attribution

The dataset preparation, channel-independent spectrogram method, Residual CNN
reference and open-set implementation originate from the work of Guanxiong
Shen, Junqing Zhang, and their co-authors. Please cite the relevant papers and
the original repository when using this code.
