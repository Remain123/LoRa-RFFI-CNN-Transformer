# LoRa RFFI: Residual CNN and Transformer Comparison

This repository contains the implementation and archived experimental results
of a Master of Science dissertation in Communications and Signal Processing.
The dissertation is primarily based on the closed-set LoRa radio-frequency
fingerprint identification (RFFI) work of Guanxiong Shen, Junqing Zhang, and
their co-authors. Their dataset, signal-processing pipeline,
channel-independent spectrogram representation, and Residual CNN are reproduced
as the reference framework. The main work of this dissertation is to replace
the original CNN classifier with four progressively refined Transformer-based
models and compare their performance with the reproduced CNN baseline under the
same closed-set task.

## Dissertation scope

The experimental work is limited to closed-set identification of 30 enrolled
LoRa transmitters. Every device appearing in the test set is also represented
in the training set. The dissertation does not implement or evaluate open-set
recognition, unseen-device rejection, or rogue-device detection.

The `Openset_RFFI_TIFS/` directory is retained from the original upstream
repository for reference and attribution only. None of its code is used in the
training, testing, model comparison, or reported results of this dissertation.

The dissertation-specific implementation extends the original
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
Openset_RFFI_TIFS/            Upstream reference only; not used in this study
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

The core LoRa RFFI framework—including the dataset, dataset-preparation
procedure, signal preprocessing, channel-independent spectrogram method, and
Residual CNN baseline—originates from the work of Guanxiong Shen, Junqing Zhang,
and their co-authors. The contribution of this dissertation is the design and
evaluation of four Transformer-based alternatives and their comparison with the
reproduced CNN baseline; it does not claim authorship of the underlying LoRa
RFFI framework. Please cite the relevant papers and the original repository
when using this code.
