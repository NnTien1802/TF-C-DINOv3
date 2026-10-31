# TF-C with Dual DINOv3 Encoders and DINOv3-Inspired Self-Supervised Losses

**Research implementation for Human Activity Recognition (HAR) and Sensor-Based Gesture Recognition (SGR)**

| Project metadata | Value |
| --- | --- |
| Project type | Academic research prototype; PyTorch command-line training pipeline |
| Research implementation | TF-C with two DINOv3 ViT-S/16 backbones and adapted DINO, iBOT, and KoLeo losses |
| Documentation revision | 2026-10-30 |
| Repository maintainer | Project repository owner (see repository history) |
| Upstream TF-C | [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining) |
| DINOv3 pretrained checkpoint | [`facebook/dinov3-vits16-pretrain-lvd1689m`](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m) |
| Verification status | **4/4 offline CPU unit tests passed; real DINOv3 end-to-end training has not yet been verified** |

> **Important distinction:** This repository adapts selected concepts from DINOv3 to TF-C and accelerometer time series. It is **not** an official Meta DINOv3 implementation, a reproduction of its full image-pretraining recipe, or a demonstrated improvement in accuracy. Performance results must be collected from actual HAR/SGR experiments before any model-performance claim is made.

## 1. Project Overview

**Problem.** Recognizing human activities and gestures from smartphone and smartwatch accelerometers requires useful representations of multivariate time-series signals. The original TF-C framework learns representations using consistency between time-domain and frequency-domain views.

**Proposed approach.** This research version retains TF-C's two-domain design but replaces both original Transformer encoders with separate pretrained **DINOv3 ViT-S/16** backbones. It also replaces the original NT-Xent-Poly training objective with a **DINOv3-inspired student–teacher objective** combining CLS-token distillation (DINO), masked patch-token distillation (iBOT), and embedding diversity regularization (KoLeo).

**Intended users.** Students, researchers, and reviewers evaluating time-series representation learning, architectural modifications, loss functions, and downstream sensor classification.

**Scope of this code package:**

- Read prepared, labeled HAR/SGR `.pt` datasets with independent train/validation/test splits.
- Encode the same three-axis signal in **time** and **frequency** branches using two independent DINOv3 backbones.
- Run self-supervised pre-training with a student and an exponential-moving-average (EMA) teacher.
- Fine-tune a classifier and evaluate HAR/SGR predictions on held-out target data.
- Save training logs, per-epoch metrics, model checkpoints, and final test metrics.
- Run synthetic offline tests without downloading DINOv3 weights.

**Out of scope of this code package:** loading raw Sensor Logger CSV/ZIP recordings directly, collecting data, a complete web/API inference service, reproducing every official DINOv3 training detail, and a completed benchmark of 10 methods.

## 2. System Overview

### 2.1 Architecture

```text
Input: 3-axis accelerometer window [B, 3, 206]
                            |
                +-----------+-----------+
                |                       |
        Time-domain signal       Magnitude FFT signal
                |                       |
        XYZ-to-image adapter      XYZ-to-image adapter
                |                       |
       DINOv3 ViT-S/16 (T)      DINOv3 ViT-S/16 (F)
                |                       |
          CLS + patch tokens      CLS + patch tokens
                |                       |
        Projector / DINO head    Projector / DINO head
          + iBOT patch head       + iBOT patch head
                |                       |
                +-----------+-----------+
                            |
                  DINO + iBOT + KoLeo
               (EMA teacher during SSL)
                            |
                   [z_time || z_freq]
                            |
                 Downstream classifier
                            |
                 HAR / SGR class logits
```

**Execution location:** training and evaluation run locally in Python on a CPU or CUDA GPU. The code package does not launch a backend or web server. No database, Node.js installation, or `.env` file is required by the current Python scripts.

### 2.2 Processing flow

1. **Data loading:** load `train.pt`, `val.pt`, and `test.pt`; preserve acceleration axes X, Y, and Z.
2. **Time view:** apply per-axis normalization and resize the one-dimensional signal into an image-like tensor.
3. **Frequency view:** calculate the magnitude of the FFT, then use the image adapter on the frequency representation.
4. **Feature extraction:** process the two images with independent, pretrained DINOv3 ViT backbones.
5. **Self-supervised pre-training:** predict teacher targets with the DINO/iBOT/KoLeo objective and update the teacher using EMA.
6. **Supervised fine-tuning:** concatenate the two 128-dimensional projected features, train a classification head, and select the checkpoint using **validation macro-F1**.
7. **Final evaluation:** load the best validation checkpoint and evaluate the target **test set once**.

The XYZ-to-image adapter repeats each one-dimensional channel over image height. It is **not** a learned image representation or a validated spectrogram transform; its suitability for accelerometer recognition remains an experimental question.

## 3. Project Structure and Code Changes

```text
TFC_DINOv3_Loss_Updated/
├── README.md
├── main.py                     # CLI, configuration, device, checkpoints
├── model.py                    # Two DINOv3 branches, adapters, projection heads
├── loss.py                     # Adapted DINO + iBOT + KoLeo objective
├── trainer.py                  # Pre-training, fine-tuning, validation, testing
├── dataloader.py               # XYZ-preserving data loader and split enforcement
├── augmentations.py            # Time jitter and frequency masking
├── requirements.txt
├── utils.py
├── config_files/
│   ├── __init__.py
│   ├── HAR_Configs.py
│   └── SGR_Configs.py
├── tests/
│   ├── test_smoke.py            # Mock ViT, gradients, teacher update, loss
│   └── test_data.py             # XYZ dimensions and split handling
├── model_original.py           # Original supplied file for comparison
├── loss_original.py
├── trainer_original.py
├── main_original.py
└── datasets/                    # CREATE locally; dataset files are not bundled
    ├── HAR/
    │   ├── train.pt
    │   ├── val.pt
    │   └── test.pt
    └── SGR/
        ├── train.pt
        ├── val.pt
        └── test.pt
```

### 3.1 Modified modules

| File | Original behavior | Modified behavior |
| --- | --- | --- |
| [`model.py`](model.py) | Two PyTorch `TransformerEncoder` branches | Two independent DINOv3 ViT-S/16 backbones; time/frequency image adapters; CLS and patch features; DINO/iBOT heads |
| [`loss.py`](loss.py) | `NTXentLoss_poly` used for TF-C contrastive training | `DINOv3AdaptedLoss`, local Sinkhorn-Knopp teacher assignments, DINO distillation, iBOT patch loss, and KoLeo |
| [`trainer.py`](trainer.py) | TF-C contrastive objective and downstream training | EMA teacher, gradient reset per batch, modified SSL objective, validation-based checkpointing, and a single final test evaluation |
| [`main.py`](main.py) | Original dataset and Transformer-specific training entry point | Explicit DINOv3 and loss CLI options, model compatibility checks, portable dataset/output paths |
| [`dataloader.py`](dataloader.py) | Prior processing could select only one input axis | Preserves all three XYZ axes and requires a separate validation split |
| [`augmentations.py`](augmentations.py) | Original augmentation implementations | Torch-native jitter and frequency masking that do not require hardcoded CUDA tensors |
| `tests/` | Not supplied in the four-file baseline | Synthetic unit tests for dimensions, loss gradients, EMA, and data boundaries |

For reviewer-friendly comparisons, original versions of the four user-supplied files are included as `*_original.py`. These snapshots are for reference, **not** runnable experiment presets. A complete original TF-C baseline requires the original repository and its own environment.

### 3.2 The implemented loss function

The self-supervised pre-training objective is:

```text
L_ssl = w_dino * [L_dino_within + w_cross * L_dino_cross]
      + w_ibot * L_ibot
      + w_koleo * L_koleo
```

- **DINO / within-domain:** an augmented student CLS representation is trained to match an unaugmented EMA teacher distribution in the same domain.
- **DINO / cross-domain:** time and frequency student CLS outputs are also matched to teacher distributions from the opposite branch.
- **iBOT:** masked student patch predictions match unmasked teacher patch targets **within the same domain**. Time and frequency patch positions are not assumed to correspond.
- **KoLeo:** regularizes student embedding diversity through nearest-neighbor distances.
- **EMA teacher:** teacher parameters are updated from student parameters; the teacher is not optimized by backpropagation.

The supervised fine-tuning objective is:

```text
L_finetune = CrossEntropy(class_logits, labels)
           + finetune_ssl_weight * L_ssl
```

The default `finetune_ssl_weight` is `0.0` (classification loss only). Optional SSL regularization can be enabled for research ablation experiments.

**Implementation differences from official DINOv3:** this package does not reproduce the exact DINOHead, large-scale distributed/multi-crop training, all official schedules, or optional Gram Loss. It uses a lightweight local student–teacher adaptation for sensor data and small batches; it should be described as *DINOv3-inspired*, not *the official DINOv3 loss implementation*.

## 4. Environment and Prerequisites

| Component | Requirement / note |
| --- | --- |
| Operating system | Windows 10/11 (instructions below); Linux is also possible but is not verified here |
| Python | **3.11.x** recommended for the existing `tfc-dinov3-py311` Conda environment |
| Deep learning framework | PyTorch with a build appropriate for the computer's CPU/CUDA setup |
| Transformers | `>=4.57,<5`, as declared in `requirements.txt` |
| Other dependencies | `huggingface_hub>=0.34`, `numpy>=1.24`, `scikit-learn>=1.3` |
| Model weights | `facebook/dinov3-vits16-pretrain-lvd1689m`; downloaded separately, not included in this repository |
| Internet access | Required for the first model download unless the weights are already cached |
| Authentication | Hugging Face login and model-access approval may be needed for the checkpoint |
| GPU | CUDA-capable GPU recommended for real training; a laptop GPU with 8 GB VRAM may be insufficient for some configurations |
| Database, Node.js, API keys, `.env` | **Not used by this Python training package** |

> **Verified test environment:** the included offline unit tests passed in this session with **Python 3.13 and PyTorch 2.10.0+cpu**. The full pretrained DINOv3 model was **not** loaded or trained in that environment because `transformers` and the actual model weights were unavailable. The recommended Windows/Python 3.11/CUDA environment must be verified on the user's machine.

### 4.1 Memory considerations

Two independent student backbones and a deep-copied EMA teacher can require **four ViT backbone instances** in memory. The backbone parameters are frozen by default to reduce training memory, but this does **not** eliminate model-weight memory or guarantee that 8 GB VRAM is sufficient.

For an initial GPU check, start with `--batch_size 2 --target_batch_size 2 --image_size 112`, and **do not** enable `--unfreeze_backbone`. If out-of-memory errors persist, reduce batch size, review GPU memory use, or move the experiment to a larger GPU. Freezing the backbone also means the original pretrained ViT weights are **not** updated during standard training; projection and prototype heads still learn.

## 5. Setup (Windows / Conda / VS Code)

Run the following commands from **Anaconda Prompt** or the **VS Code integrated terminal**. All commands in subsequent sections assume the terminal is in the project directory containing `main.py`.

### Step 1 — Obtain the project and open its folder

Download and extract the supplied source-code archive, or clone the repository **after uploading this modified code to GitHub**:

```powershell
# Replace the URL with the repository URL containing this modified version.
git clone <YOUR_MODIFIED_REPOSITORY_URL>
cd <YOUR_REPOSITORY_DIRECTORY>
```

If using the downloaded ZIP, extract it and run `cd` into `TFC_DINOv3_Loss_Updated`. Verify that `main.py`, `model.py`, `loss.py`, and `requirements.txt` are visible.

### Step 2 — Activate or create a Conda environment

```powershell
# Use the existing environment if it has already been created.
conda activate tfc-dinov3-py311
python --version
```

If that environment does not exist, create it first:

```powershell
conda create -n tfc-dinov3-py311 python=3.11 -y
conda activate tfc-dinov3-py311
python --version
```

**Expected:** the reported Python version begins with `3.11` and the terminal prompt shows `(tfc-dinov3-py311)`.

### Step 3 — Install dependencies

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU acceleration, ensure the installed PyTorch build matches your NVIDIA driver and CUDA runtime; follow the [official PyTorch installation selector](https://pytorch.org/get-started/locally/) rather than assuming one CUDA wheel works on every machine.

Verify imports and CUDA availability:

```powershell
python -c "import torch, transformers; print('torch:', torch.__version__); print('transformers:', transformers.__version__); print('CUDA available:', torch.cuda.is_available())"
```

**Expected:** the command prints both package versions without an import error. On a properly configured CUDA computer, `CUDA available: True` is expected. CPU-only tests can still run when it prints `False`.

### Step 4 — Obtain DINOv3 checkpoint access

1. Open the [DINOv3 ViT-S/16 model page](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).
2. Accept applicable access terms and obtain permission if the model is gated.
3. Authenticate locally:

```powershell
hf auth login
```

**Expected:** authentication succeeds, and a subsequent `from_pretrained` request can access the checkpoint. Do **not** include access tokens in this README, Git commits, or screenshots. Downloaded pretrained weights are cached separately by Hugging Face; they are not contained in the project ZIP.

### Step 5 — Prepare the datasets

Download or copy the prepared `HAR.zip` and `SGR.zip` datasets. Create a `datasets` folder next to `main.py`, and extract each archive so that the actual files are located at:

```text
datasets/HAR/train.pt
datasets/HAR/val.pt
datasets/HAR/test.pt
datasets/SGR/train.pt
datasets/SGR/val.pt
datasets/SGR/test.pt
```

**PowerShell example** (assuming `HAR.zip` and `SGR.zip` are in the project directory):

```powershell
New-Item -ItemType Directory -Path datasets -Force
Expand-Archive -Path HAR.zip -DestinationPath datasets -Force
Expand-Archive -Path SGR.zip -DestinationPath datasets -Force
Test-Path datasets/HAR/train.pt
Test-Path datasets/SGR/val.pt
```

**Expected:** both `Test-Path` commands return `True`. If you obtain the datasets from [the project's Kaggle dataset](https://www.kaggle.com/datasets/nguynngctin1881/dataset-tf-c), download the prepared `.pt` splits and arrange them in the same directory layout.

Each `.pt` split must contain a dictionary with these keys:

```python
{
    "samples": Tensor[N, 3, 206],  # or [N, 206, 3]
    "labels": Tensor[N]             # integer class IDs starting at 0
}
```

The loader transposes `[N, 206, 3]` if necessary and pads/truncates signals to the configured 206 samples. **Do not replace `val.pt` with `test.pt`:** the loader requires an independent validation file. Avoid placing recordings from the same acquisition session across train/validation/test when preparing new splits.

**Datasets used in the project:**

| Dataset | Samples / recordings | Classes | Role |
| --- | ---: | --- | --- |
| HAR (prepared public data) | 10,369 windows | Walking, Stairs, Sitting, Standing | Source or downstream classification; prepared `.pt` splits |
| SGR (prepared public data) | 1,200 windows | Typing, Teeth, Drinking, Writing, Clapping, Folding | Source or downstream classification; prepared `.pt` splits |
| Private HAR (Sensor Logger) | 12 recording sessions; 19,795 raw measurements | Walking, Stairs, Sitting, Standing | Separate exploratory/external data; **not directly loaded by this training script** |

The private CSV recordings require an explicit preprocessing, windowing, labeling, and leakage-safe splitting pipeline before they can be passed to this loader. The public datasets and private recordings should not be represented as the same evaluation split.

## 6. Run & Verify

### Step 6 — Run offline unit tests first

From the directory containing `main.py`:

```powershell
python -m unittest discover -s tests -v
```

**Expected result:** four tests pass, and the command ends with output similar to:

```text
Ran 4 tests in ...s
OK
```

These tests use a lightweight **mock ViT**; they check tensor shapes, gradients, EMA logic, and dataset handling. Passing them **does not establish** that the real Hugging Face DINOv3 checkpoint loads or that the model achieves good classification accuracy.

### Step 7 — Run a short real-model pre-training check

```powershell
python main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 1 --image_size 112 --subset
```

**Expected behavior (if dependencies, checkpoint access, GPU memory, and dataset paths are valid):** the model initializes, runs one pre-training epoch on a subset of HAR, logs the loss components, and saves:

```text
experiments_logs/HAR_2_SGR/dinov3_loss/pre_train_seed_42/
├── train.log
├── epoch_metrics.csv
└── saved_models/
    └── ckp_last.pt
```

The per-epoch CSV should contain `loss`, `dino_within`, `dino_cross`, `ibot`, and `koleo`. The exact numeric losses depend on the actual run; no example loss value should be presented as a verified result.

### Step 8 — Run full pre-training

After the short test succeeds, run the full data split **without `--subset`**:

```powershell
python main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112
```

This writes to the same pre-training output directory as Step 7 and replaces its final checkpoint. Use `--run_description` to store alternative experiments separately.

### Step 9 — Fine-tune and evaluate the target dataset

```powershell
python main.py --training_mode fine_tune_test --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112
```

The command loads the pretrained checkpoint from Step 8 and fine-tunes on **SGR train**, selects the best checkpoint using **SGR validation macro-F1**, and evaluates **SGR test only after selecting that checkpoint**.

**Expected files:**

```text
experiments_logs/HAR_2_SGR/dinov3_loss/fine_tune_test_seed_42/
├── train.log
├── epoch_metrics.csv
├── test_metrics.txt
└── saved_models/
    └── ckp_best.pt
```

`test_metrics.txt` contains held-out target metrics including accuracy, macro precision, macro recall, macro F1, and confusion matrix. **No real test metrics have been reported for this updated model yet.**

### Step 10 — Optional experiment configurations

```powershell
# Optional: add self-supervised loss during fine-tuning (extra computation).
python main.py --training_mode fine_tune_test --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112 --finetune_ssl_weight 0.1

# Optional: use a separate experiment directory.
python main.py --training_mode pre_train --pretrain_dataset HAR --target_dataset SGR --batch_size 2 --target_batch_size 2 --epochs 3 --image_size 112 --run_description exp_dino_ibot_koleo
```

For the first optional fine-tune example, the matching pre-training checkpoint must already exist under the same `run_description`, model ID, image size, prototype count, and seed. Additional runs can require more GPU memory and time.

## 7. Usage: Command-Line Interface

The current package provides a **CLI**, not a REST API or web interface.

```powershell
python main.py --help
```

Selected flags (all implemented in [`main.py`](main.py)):

| Option | Default | Purpose |
| --- | --- | --- |
| `--training_mode` | `pre_train` | `pre_train` or `fine_tune_test` |
| `--pretrain_dataset` | `HAR` | Source dataset folder |
| `--target_dataset` | `SGR` | Target dataset folder |
| `--dataset_root` | `datasets` | Parent directory containing dataset folders |
| `--logs_save_dir` | `experiments_logs` | Output folder |
| `--run_description` | `dinov3_loss` | Experiment identifier (folder name) |
| `--device` | `auto` | `auto`, `cpu`, or `cuda` |
| `--seed` | `42` | Random seed |
| `--epochs` | `3` | Epochs in the selected training stage |
| `--batch_size` | `8` | Source pre-training batch size |
| `--target_batch_size` | `8` | Target dataset batch size |
| `--lr` | `1e-4` | Optimizer learning rate |
| `--model_id` | `facebook/dinov3-vits16-pretrain-lvd1689m` | HF checkpoint ID |
| `--image_size` | `112` | Resized signal-image height and width; must be divisible by patch size 16 |
| `--prototypes` | `256` | Number of distillation prototypes |
| `--teacher_temp` | `0.04` | Teacher distribution temperature |
| `--student_temp` | `0.1` | Student temperature |
| `--ema_momentum` | `0.996` | EMA teacher update momentum |
| `--mask_ratio` | `0.3` | Fraction of masked patch tokens |
| `--dino_weight` | `1.0` | CLS distillation weight |
| `--ibot_weight` | `1.0` | Masked patch distillation weight |
| `--koleo_weight` | `0.1` | Diversity loss weight |
| `--cross_domain_weight` | `0.25` | Cross time/frequency distillation weight |
| `--finetune_ssl_weight` | `0.0` | Optional SSL component in supervised fine-tuning |
| `--unfreeze_backbone` | Disabled | Allow ViT backbone weight updates; high GPU memory demand |
| `--subset` | Disabled | Use a small source/target training subset for debugging |

**Model interface:** the core model takes time-domain and FFT-domain tensors and returns four tensors:

```python
h_t, z_t, h_f, z_f = model(time_tensor, frequency_tensor)
# With ViT-S/16: h_t, h_f are [B, 384]; z_t, z_f are [B, 128].
# The downstream classifier consumes concat(z_t, z_f) -> [B, num_classes].
```

Both encoders are initialized from the **same pretrained checkpoint**, but are separate model instances. The full forward pass also exposes patch features for training via `forward_ssl()`.

## 8. AI Model and Evaluation Protocol

### 8.1 Model card

| Field | Description |
| --- | --- |
| Model name | TF-C Dual-DINOv3 (research adaptation) |
| Backbone | Two independent `DINOv3ViTModel` instances initialized from Meta ViT-S/16 |
| Model weights | Not distributed with this repository; fetched from Hugging Face |
| Input | XYZ acceleration windows: `[B, 3, 206]`; prepared FFT magnitude internally in the loader |
| Backbone input | Image-like `[B, 3, image_size, image_size]` for each domain |
| Backbone output | CLS feature vectors (ViT-S/16: 384 dims) plus patch tokens |
| Projected output | Two 128-dimensional embeddings |
| Classification output | Unnormalized logits `[B, num_classes_target]` |
| Prediction | `argmax(logits, dim=-1)`; confidence calibration not yet implemented |
| Pre-training objective | Adapted DINO + iBOT + KoLeo |
| Fine-tuning objective | Cross-Entropy, optional SSL regularization |
| Model artifact | `saved_models/ckp_last.pt` (pre-training); `saved_models/ckp_best.pt` (fine-tuning) |
| Versioning | Record Git commit, dataset version, `model_id`, run description, seed, and CLI arguments per experiment |
| Main limitations | Image-domain transfer uncertainty; frozen pretrained backbones by default; small-batch Sinkhorn; incomplete official DINOv3 recipe |

The scripts do not currently export ONNX, serve a `/predict` endpoint, return calibrated confidence, or implement a human-review fallback. These should not be claimed as implemented product features. If connecting a separate web demo, verify that it loads **this updated checkpoint**, rather than an older TF-C model.

### 8.2 Train/validation/test boundaries

- **Pre-training:** source **train** split only; labels are not used to supervise the DINO/iBOT objective.
- **Fine-tuning:** target **train** split only.
- **Model selection:** target **validation** macro-F1 only.
- **Final report:** target **test** metrics after selecting a checkpoint; never tune hyperparameters using the test set.
- **Private external evaluation:** create a separate, documented preprocessing/splitting process; keep recording-level groups together to prevent data leakage.

### 8.3 Recommended experiment table (to be completed after real runs)

| Experiment | Encoder | SSL objective | Test macro-F1 | Accuracy | p50/p95 latency | Status |
| --- | --- | --- | --- | --- | --- | --- |
| E0 | Original TF-C Transformer | Original NT-Xent-Poly | — | — | — | Not run in this package |
| E1 | Original TF-C Transformer | Modified loss | — | — | — | Ablation requires verified integration |
| E2 | Dual DINOv3 | Original TF-C loss | — | — | — | Ablation requires verified integration |
| E3 | Dual DINOv3 | DINO + iBOT + KoLeo | — | — | — | Code prepared; real checkpoint test pending |

**Do not fill these columns with synthetic test outputs.** Record actual dataset version and split, GPU/CPU hardware, PyTorch/Transformers versions, hyperparameters, random seed, model file size, training duration, measured inference latency, and a confusion matrix. The course guidance also calls for error analysis and traceable evidence when reporting experiments.

A systematic extension could test the loss weights individually, freeze/unfreeze settings, image sizes, and model size/latency, but such experiments are **proposals**, not completed results.

### 8.4 Known limitations and risks

1. Pretrained DINOv3 visual representations may not transfer effectively to repeated-row accelerometer images.
2. Image size, FFT handling, and small-batch Sinkhorn assignments can affect representation quality.
3. Frozen backbones limit adaptation; unfrozen backbones may exceed laptop GPU memory.
4. Two student encoders plus an EMA teacher increase RAM/VRAM usage and inference setup cost.
5. Distillation and iBOT heads are lighter research substitutes, not exact copies of the official Meta modules.
6. The provided unit tests use a mock backbone, not real pretrained weights.
7. Preprocessed public HAR/SGR datasets are not automatically equivalent to the private raw Sensor Logger dataset.
8. The current CLI does not include a standalone serving API, confidence threshold, error fallback, model export, or end-user acceptance test.

## 9. Troubleshooting

| Symptom | Possible cause | Recommended action |
| --- | --- | --- |
| `ModuleNotFoundError: transformers` | Dependencies missing or wrong Conda environment | Activate `tfc-dinov3-py311`; run `python -m pip install -r requirements.txt` |
| `ImportError` for `DINOv3ViTModel` | Transformers build lacks the required model class | Check `transformers.__version__` and upgrade to a compatible version specified in requirements |
| `401/403`, gated repository, or authentication error | Hugging Face account lacks model access | Request/accept access on the model page; run `hf auth login`; verify cached credentials |
| CUDA unavailable | CPU PyTorch build, driver mismatch, or missing GPU support | Run the import/CUDA check; install a compatible official PyTorch CUDA build |
| `CUDA out of memory` | Multiple ViT student/teacher copies do not fit | Try batch size 2 or 1, image size 112, frozen backbones, close other GPU processes, or use a larger GPU |
| `Missing dataset split` | ZIP extracted into wrong folder or file omitted | Check exact paths `datasets/HAR/train.pt`, etc., and `--dataset_root` |
| `Expected independent validation set` | `val.pt` is missing | Create a real separate validation split from appropriate labeled data; do **not** reuse test data |
| `Expected ... 3 channels` | Data contains only one axis or wrong tensor layout | Prepare XYZ inputs `[N,3,L]` or `[N,L,3]`; verify sample acquisition and feature selection |
| Checkpoint not found in fine-tuning | Pre-training was not completed with matching settings | Run pre-training first; check `run_description`, dataset names, seed, and output paths |
| `Checkpoint architecture differs` | Model ID, image size, or prototype count changed | Use the checkpoint's exact architecture settings or generate a new matching checkpoint |
| Loss becomes non-finite | Numerical instability, unsuitable hyperparameters, or unexpected data | Inspect input values and augmentation, reduce learning rate, log loss components, and stop the run |
| Tests pass but real model fails | Offline tests only use a mock ViT | Check model access, Transformers compatibility, real input dimensions, device memory, and full traceback |

For debugging, include the exact command, error traceback, operating system, Python/PyTorch/Transformers versions, GPU model and VRAM, and sanitized output logs. Do not share Hugging Face authentication tokens.

## 10. Reproducibility and Reviewer Checklist

Use this checklist before submitting the project:

- [ ] This README identifies the **problem, user, input, output, and current scope**.
- [ ] Environment versions and external model access requirements are documented.
- [ ] The modified repository can be cloned or extracted without missing Python source files.
- [ ] Dataset directories contain separate **train**, **validation**, and **test** splits.
- [ ] `python -m unittest discover -s tests -v` passes on the reviewer's machine.
- [ ] A real-model subset pre-training run completes and writes its expected checkpoint.
- [ ] Fine-tuning completes using a matching pre-training checkpoint.
- [ ] A separate held-out test run produces `test_metrics.txt` and confusion matrix.
- [ ] Changes to **both encoders** and the **loss function** are visible in the code comparison table.
- [ ] Actual metrics, error examples, model size, inference latency, hardware, and limitations are reported without fabricated measurements.
- [ ] The Git commit, dataset version, configuration, and model artifact are traceable.
- [ ] Any claimed external demo is explicitly connected to the updated model (otherwise presented as a separate prototype).

**Current evidence:** four local CPU mock-backbone/unit tests passed. Real DINOv3 checkpoint loading, HAR/SGR training metrics, latency benchmarks, and model export/deployment remain **unverified**. These are open validation tasks rather than completed deliverables.

## 11. References and Attribution

1. **Original TF-C repository:** [mims-harvard/TFC-pretraining](https://github.com/mims-harvard/TFC-pretraining).
2. **TF-C publication:** *Self-Supervised Contrastive Pre-Training for Time Series via Time-Frequency Consistency*, NeurIPS 2022 — [paper](https://arxiv.org/abs/2206.08496).
3. **DINOv3 implementation:** [facebookresearch/dinov3](https://github.com/facebookresearch/dinov3) — [official loss modules](https://github.com/facebookresearch/dinov3/tree/main/dinov3/loss) and [student–teacher architecture](https://github.com/facebookresearch/dinov3/blob/main/dinov3/train/ssl_meta_arch.py).
4. **Pretrained DINOv3 weights:** [Meta DINOv3 ViT-S/16 on Hugging Face](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m); follow the checkpoint's access terms and license.
5. **Course documentation criteria:** *AI & Data Science Special Topics* lecture slides by Dat Tran, **Session 10: How to Write a README**, PDF pages 143–158. This README follows the lecture's sections: Project Overview, System Overview, Environment, Setup, Run & Verify, Usage, AI Model, Troubleshooting, and explicit expected outputs.

> **Academic attribution:** The upstream TF-C framework and official DINOv3 model belong to their respective authors. The present code is a research adaptation for course experimentation and should not be presented as either team's official implementation.
