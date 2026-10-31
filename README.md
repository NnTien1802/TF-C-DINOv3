# TF-C-DINOv3: Time–Frequency Representation Learning with Dual DINOv3 Encoders

**An academic research implementation for Human Activity Recognition (HAR) and Sensor-Based Gesture Recognition (SGR) from three-axis accelerometer data.**

[![Python](https://img.shields.io/badge/Python-3.11-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
[![Research](https://img.shields.io/badge/Status-Research%20Prototype-lightgrey)](https://github.com/NnTien1802/TF-C-DINOv3)

**Repository:** https://github.com/NnTien1802/TF-C-DINOv3  
**Based on:** [TF-C Pretraining](https://github.com/mims-harvard/TFC-pretraining)  
**Pretrained backbone:** [DINOv3 ViT-S/16](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m)

> **Implementation status:** The modified source code and offline unit tests are available. Offline tests use a mock ViT backbone; successfully training and evaluating the real DINOv3 checkpoint on HAR/SGR has **not yet been established**. No accuracy, F1-score, or deployment benchmark is claimed here.
>
> **Research attribution:** This is a **DINOv3-inspired adaptation of TF-C**, not the official DINOv3 training implementation from Meta and not a full reproduction of its large-scale pretraining procedure.

## 1. Project Overview

### 1.1 Problem and objective

Accelerometer-based activity and gesture recognition involves classifying sequences of motion measurements from three sensor axes: **X, Y, and Z**. The original **Time–Frequency Contrastive Learning (TF-C)** framework learns features by contrasting time-domain and frequency-domain representations.

This project investigates two architectural modifications:

1. **Replace both TF-C encoders** with two separate pretrained **DINOv3 ViT-S/16** backbones: one for time-domain signals and one for frequency-domain features.
2. **Replace the original NT-Xent-Poly-based objective** with a lightweight **DINOv3-inspired self-supervised loss** combining DINO CLS-token distillation, iBOT masked-patch distillation, and KoLeo regularization.

The intended outcome is a reproducible experimental pipeline for comparing representation quality and downstream classification, not a production-ready mobile or web application.

### 1.2 Project scope

The current Python project supports:

- Loading prepared HAR/SGR datasets saved as PyTorch `.pt` files.
- Preserving all **three accelerometer channels** in the loading and augmentation pipeline.
- Generating time-domain and FFT-magnitude frequency-domain views.
- Self-supervised pretraining with a student network and an exponential moving average (**EMA**) teacher.
- Supervised fine-tuning, validation-based model selection, and held-out test evaluation.
- Saving checkpoints, training logs, and evaluation metrics.
- Running offline unit tests with a mock vision backbone.

Raw Sensor Logger CSV recordings, a web interface, a REST API, ONNX export, and a complete 10-method benchmark are **not implemented in this training package**.

## 2. System Overview

### 2.1 Architecture and data flow

```text
Accelerometer window [B, 3, 206]
                  |
           Data preparation
                  |
        +---------+---------+
        |                   |
    Time signal       FFT magnitude
        |                   |
   XYZ-to-image        XYZ-to-image
      adapter             adapter
        |                   |
  DINOv3 ViT-S/16    DINOv3 ViT-S/16
   (Time branch)    (Frequency branch)
        |                   |
  CLS/patch tokens   CLS/patch tokens
        |                   |
  Projection heads   Projection heads
        |                   |
        +---------+---------+
                  |
        DINO + iBOT + KoLeo
       (student / EMA teacher)
                  |
      Concatenated embeddings
                  |
         Classifier head
                  |
          HAR / SGR logits
```

**Time branch:** each of the X/Y/Z signals is normalized, resized, and repeated vertically to form a three-channel image-like tensor.  
**Frequency branch:** the magnitude of the FFT is transformed using `log1p`, normalized, and adapted to an image-like tensor.  
**Backbones:** the time and frequency branches use **independent model instances**, initially loaded from the same pretrained DINOv3 weights.  
**Classifier:** concatenates two 128-dimensional projected features and returns class logits.

The image adapters are simple signal-to-image transformations. They are **not** trained spectrogram generators, and their suitability for DINOv3 must be established through real experiments.

### 2.2 Modified self-supervised loss

The adapted loss used in pretraining is:

```text
L_SSL = w_dino * (L_DINO_within + w_cross * L_DINO_cross)
      + w_ibot * L_iBOT
      + w_koleo * L_KoLeo
```

| Component | Function |
| --- | --- |
| DINO within-domain | Matches augmented student CLS-token predictions to EMA-teacher targets within each domain. |
| DINO cross-domain | Aligns student representations with teacher predictions from the other domain. |
| iBOT | Matches masked student patch predictions to unmasked teacher patch targets within the same domain. |
| KoLeo | Encourages diversity among normalized student embeddings. |
| EMA teacher | Updates teacher parameters from student parameters without backpropagating through the teacher. |

The teacher targets use a **local Sinkhorn–Knopp** procedure. This implementation does **not** reproduce Meta's complete distributed DINOv3 recipe or optional Gram Loss.

Fine-tuning uses supervised **Cross-Entropy Loss** by default. An optional self-supervised regularizer is controlled by `--finetune_ssl_weight` (default: `0.0`).

## 3. Project Structure and Modified Files

The repository currently contains two separate code directories: `code_goc/` for the original TF-C reference files and `code_da_sua/` for the modified implementation. The directory tree below reflects the files visible in the GitHub repository on the `main` branch.

```text
TF-C-DINOv3/
├── README.md                    # Project documentation
├── code_goc/                   # Original TF-C source files
│   ├── loss.py                 # Original contrastive loss functions
│   ├── main.py                 # Original training entry point
│   ├── model.py                # Original TF-C encoders and classifier
│   └── trainer.py              # Original training and evaluation routines
└── code_da_sua/               # Modified TF-C implementation
    ├── loss.py                 # DINOv3-inspired loss components
    ├── main.py                 # Updated training entry point
    ├── model.py                # Dual DINOv3 encoder implementation
    └── trainer.py              # Updated training and evaluation routines
```

**Repository layout note:** Only the files shown above are present in this GitHub directory view. Additional dependencies such as `dataloader.py`, `augmentations.py`, `utils.py`, `config_files/`, datasets, and environment files are **not shown in the repository**. Running the training pipeline may require these files to be added or restored separately; their presence should not be assumed.

### 3.1 Summary of code changes

| File | Modification |
| --- | --- |
| `model.py` | Replaced the two original Transformer encoders with time/frequency DINOv3 ViT-S/16 branches; added adapters, projection heads, and token-level outputs. |
| `loss.py` | Added a compact DINOv3-inspired student–teacher objective using DINO, iBOT, KoLeo, and local Sinkhorn assignments. |
| `trainer.py` | Added an EMA teacher and new SSL training flow; updates gradients per batch; selects the final model using validation macro-F1. |
| `main.py` | Added explicit command-line controls, consistent checkpoint paths, source/target dataset options, and training configuration. |
| `dataloader.py` | Preserves all three sensor axes and requires a separate validation split rather than reusing test data. |
| `augmentations.py` | Uses Torch-based time/frequency augmentations without hardcoded CUDA allocation. |
| `tests/` | Provides offline unit tests that do not require downloading the real DINOv3 model. |

The `*_original.py` files are **reference snapshots** for reviewers and are not executed by the updated training commands. They do not form a ready-to-run original TF-C baseline on their own.

### 3.2 Datasets

| Dataset | Description | Prepared splits |
| --- | --- | --- |
| HAR | Smartphone accelerometer activity windows; 10,369 samples across Walking, Stairs, Sitting, and Standing. | `train.pt`, `val.pt`, `test.pt` |
| SGR | Smartwatch accelerometer gesture windows; 1,200 samples across Typing, Teeth, Drinking, Writing, Clapping, and Folding. | `train.pt`, `val.pt`, `test.pt` |
| Private HAR | 12 self-recorded Sensor Logger sessions with 19,795 raw accelerometer measurements. | Raw CSV data; **not directly consumed** by `dataloader.py` |

**Dataset location:** [TF-C: Human Activity and Gesture Recognition Data on Kaggle](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c).  
**Public EDA notebook:** [EDA_dataset](https://www.kaggle.com/code/nguynngctin1881/eda-dataset).

The loader expects each `.pt` split to contain a dictionary with `samples` and `labels`, where `samples` has shape `[N, 3, L]` or `[N, L, 3]`. Each signal is trimmed or zero-padded to a length of 206. The dataset archives may also contain `README.md`, label mappings, and split summaries; these metadata files are optional for the loader but useful for documentation.

Do **not** mix the private raw data into the published test metrics without defining a separate preprocessing and recording-level split strategy.

## 4. Environment and Prerequisites

| Item | Requirement |
| --- | --- |
| Operating system | Windows 10/11 recommended for the instructions below |
| Python | Python **3.11.x** |
| Framework | PyTorch; install a compatible CPU or CUDA build |
| Transformers | `>=4.57,<5`, as specified by `requirements.txt` |
| Other Python packages | `huggingface_hub`, `numpy`, `scikit-learn` |
| Pretrained model | `facebook/dinov3-vits16-pretrain-lvd1689m` |
| Hardware | CPU for lightweight tests; CUDA GPU recommended for DINOv3 training |
| Storage | Space for both the pretrained checkpoint and training outputs |
| Internet access | Required for the initial model download unless already cached |

The default configuration freezes the pretrained DINOv3 backbone parameters. This reduces gradient memory, but the project may still instantiate **two student backbones and an EMA teacher copy**; training can exceed the capacity of an 8 GB GPU. Use small batches for initial checks.

### 4.1 Confirm the Python environment

```powershell
python --version
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

**Expected:** Python reports version `3.11.x`; PyTorch imports successfully. CUDA is `True` only when a compatible NVIDIA/PyTorch CUDA installation is available.

## 5. Setup

Run the following commands in **Anaconda Prompt**, **PowerShell**, or the **VS Code terminal**. Python-based archive extraction is used below to work in both CMD and PowerShell.

### Step 1 — Clone the repository

```powershell
git clone https://github.com/NnTien1802/TF-C-DINOv3.git
cd TF-C-DINOv3
```

**Expected:** the working directory contains `main.py`, `model.py`, `loss.py`, and `requirements.txt`. If those files are inside a nested source folder, open that folder instead before continuing.

### Step 2 — Create or activate the Conda environment

If your environment already exists:

```powershell
conda activate tfc-dinov3-py311
```

Otherwise:

```powershell
conda create -n tfc-dinov3-py311 python=3.11 -y
conda activate tfc-dinov3-py311
```

Check the active interpreter:

```powershell
python --version
python -c "import sys; print(sys.executable)"
```

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If CUDA is required, install the PyTorch wheel selected for your computer at [pytorch.org](https://pytorch.org/get-started/locally/) and confirm `torch.cuda.is_available()` returns `True`.

Verify the key imports:

```powershell
python -c "import torch, transformers, sklearn; print('PyTorch:', torch.__version__); print('Transformers:', transformers.__version__)"
```

### Step 4 — Obtain DINOv3 checkpoint access

The project loads the following Hugging Face model on the first real training run:

```text
facebook/dinov3-vits16-pretrain-lvd1689m
```

Open the [model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), follow its access terms, and authenticate if required:

```powershell
hf auth login
```

**Expected:** the login succeeds and your account can access the model weights. Do not commit access tokens or cached model files to GitHub.

### Step 5 — Prepare the HAR and SGR datasets

Download the prepared datasets from the [Kaggle dataset page](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c). From the folder containing `main.py`, create the local data directory and extract `HAR.zip` and `SGR.zip` into it:

```powershell
python -c "from pathlib import Path; import zipfile; root=Path('datasets'); root.mkdir(exist_ok=True); [zipfile.ZipFile(name).extractall(root) for name in ('HAR.zip','SGR.zip')]"
```

The ZIP archives already contain top-level `HAR/` and `SGR/` folders. **Do not create an extra nested `HAR/HAR` or `SGR/SGR` level.**

Verify the split files:

```powershell
python -c "from pathlib import Path; print(all((Path('datasets')/d/(split+'.pt')).is_file() for d in ('HAR','SGR') for split in ('train','val','test')))"
```

**Expected:** the command prints `True`.

## 6. Run and Verify

### Step 1 — Run the offline unit tests

```powershell
python -m unittest discover -s tests -v
```

**Expected:** four unit tests finish with `OK` when the source bundle and dependencies are installed correctly. These tests use a **mock backbone**; they do not test the downloaded DINOv3 checkpoint.

### Step 2 — Run a short DINOv3 pretraining check

```powershell
python main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 1 --image_size 112 --subset
```

**Expected, if model access, memory, and data loading succeed:** one pretraining epoch is logged and a checkpoint is saved at:

```text
experiments_logs/HAR_2_SGR/dinov3_loss/pre_train_seed_42/saved_models/ckp_last.pt
```

The same experiment directory also contains `train.log` and `epoch_metrics.csv` with the overall loss and its DINO/iBOT/KoLeo components.

### Step 3 — Run pretraining without the debug subset

```powershell
python main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112
```

This command uses the full source training split and writes a checkpoint to the same experiment directory. Use `--run_description` to keep separate runs from overwriting each other's artifacts.

### Step 4 — Fine-tune and evaluate on SGR

```powershell
python main.py --training_mode fine_tune_test --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112
```

**Expected:** the script loads the matching HAR pretraining checkpoint, fine-tunes on the **SGR training split**, selects the best epoch using **SGR validation macro-F1**, and evaluates the **SGR test split** after checkpoint selection.

```text
experiments_logs/HAR_2_SGR/dinov3_loss/fine_tune_test_seed_42/
├── train.log
├── epoch_metrics.csv
├── test_metrics.txt
└── saved_models/
    └── ckp_best.pt
```

The test report includes accuracy, precision, recall, macro-F1, and a confusion matrix. **No measured performance value is asserted in this README.**

### Step 5 — Inspect the command-line options

```powershell
python main.py --help
```

**Expected:** a list of supported arguments is printed without starting training.

## 7. Usage and Experiment Configuration

This repository provides a **Python command-line training and evaluation pipeline**. It does not currently expose an HTTP endpoint or a web application.

| Argument | Default | Description |
| --- | --- | --- |
| `--training_mode` | `pre_train` | `pre_train` or `fine_tune_test` |
| `--pretrain_dataset` | `HAR` | Source dataset directory |
| `--target_dataset` | `SGR` | Target dataset directory |
| `--dataset_root` | `datasets` | Parent folder containing the dataset splits |
| `--logs_save_dir` | `experiments_logs` | Parent directory for outputs |
| `--run_description` | `dinov3_loss` | Experiment name |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--seed` | `42` | Random seed |
| `--epochs` | `3` | Number of training epochs |
| `--batch_size` | `8` | Source pretraining batch size |
| `--target_batch_size` | `8` | Target training/validation/test batch size |
| `--lr` | `1e-4` | AdamW learning rate |
| `--model_id` | `facebook/dinov3-vits16-pretrain-lvd1689m` | Pretrained checkpoint ID |
| `--image_size` | `112` | Input image size; divisible by ViT patch size 16 |
| `--prototypes` | `256` | Number of student/teacher prototypes |
| `--teacher_temp` | `0.04` | Teacher temperature |
| `--student_temp` | `0.1` | Student temperature |
| `--ema_momentum` | `0.996` | EMA teacher update factor |
| `--mask_ratio` | `0.3` | Masked patch proportion |
| `--dino_weight` | `1.0` | DINO loss weight |
| `--ibot_weight` | `1.0` | iBOT loss weight |
| `--koleo_weight` | `0.1` | KoLeo loss weight |
| `--cross_domain_weight` | `0.25` | Cross-domain DINO alignment weight |
| `--finetune_ssl_weight` | `0.0` | Optional SSL loss weight during fine-tuning |
| `--unfreeze_backbone` | Off | Train DINOv3 backbone parameters; requires more VRAM |
| `--subset` | Off | Use a small training subset for debugging |

**Core model input:** time and FFT-domain tensors of shape `[B, 3, 206]`.  
**Representation output:** two 128-dimensional projected embeddings.  
**Classifier output:** logits of shape `[B, number_of_target_classes]`.

For controlled research comparisons, change **one parameter at a time** and record the dataset version, split, model checkpoint, random seed, hardware, validation metrics, final test metrics, and measured inference latency. Use a different `--run_description` for each experimental configuration.

## 8. Troubleshooting

| Problem | Likely reason | Recommended solution |
| --- | --- | --- |
| `ModuleNotFoundError` for a project dependency | Missing package or wrong environment | Activate `tfc-dinov3-py311`; rerun `python -m pip install -r requirements.txt`. |
| `DINOv3ViTModel` cannot be imported | Unsupported Transformers version | Check `transformers.__version__` and the constraints in `requirements.txt`. |
| Hugging Face authentication or access error | Checkpoint access not granted or login expired | Review the checkpoint's access requirements and run `hf auth login`. |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch package or incompatible driver | Install a compatible CUDA-enabled PyTorch build and verify the NVIDIA driver. |
| CUDA out-of-memory | Multiple DINOv3 student/teacher instances | Start with batch size 2 and image size 112, keep backbones frozen, or use a larger GPU. |
| `Missing dataset split` | Incorrect extraction directory | Verify `datasets/HAR/train.pt`, `datasets/SGR/val.pt`, and the other split files. |
| `Expected ... 3 channels` | Incorrect input shape or missing sensor axes | Prepare accelerometer data as `[N,3,L]` or `[N,L,3]`. |
| `Expected independent validation set` | `val.pt` is missing | Prepare a genuine validation split; never substitute the test set. |
| Pretrained checkpoint not found | Pretraining not completed or different experiment name | Run pretraining first with the same source/target, seed, and `--run_description`. |
| `Checkpoint architecture differs` | Model ID, image size, or prototype count changed | Reuse the checkpoint's original settings or train a matching checkpoint. |
| Unit tests pass but real training fails | Mock-backbone tests do not cover real DINOv3 | Check checkpoint access, actual tensor dimensions, model compatibility, and VRAM. |

When reporting an issue, include the exact command, full traceback, relevant library versions, GPU model and VRAM, and non-sensitive logs. **Never share authentication tokens.**

## 9. References and Attribution

1. **TF-C original implementation:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining).
2. **TF-C paper:** [Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency (NeurIPS 2022)](https://arxiv.org/abs/2206.08496).
3. **DINOv3 official implementation:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3).
4. **DINOv3 pretrained ViT-S/16 checkpoint:** [Hugging Face model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
5. **Course documentation:** *Artificial Intelligence and Data Science Special Topics*, instructor Dat Tran, **Session 10 — How to Write a README**, pages 143–158 of the supplied course slides. The organization emphasizes project purpose, repository structure, concrete installation steps, expected results, and troubleshooting.

This repository is an independent academic adaptation. The underlying TF-C framework and DINOv3 pretrained models remain attributable to their respective authors and are subject to their applicable licensing and access terms.
