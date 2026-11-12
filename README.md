# CT-OBCR

Project repository for:

**An Efficient CNN-Transformer Hybrid with Transferability-guided Token Merging for Oracle Bone Character Recognition**

## Overview

CT-OBCR is a CNN–Transformer framework for cross-domain oracle bone
character recognition. It transfers supervision from labeled handprinted
characters to unlabeled scanned characters.

The framework combines:
- Transferability-guided token merging.
- Wavelet-based multi-scale feature extraction.
- Bidirectional cross-stream interaction.
- Pseudo-label selection discriminative clustering.

## Datasets

The following datasets are used to evaluate our framework. Dataset resources and the corresponding papers are provided below.

| Dataset | Dataset Resources | Reference Paper |
| --- | --- | --- |
| Oracle-241 | [STSN Repository](https://github.com/wm-bupt/STSN) | [Paper](https://ieeexplore.ieee.org/abstract/document/9757826) |
| Digit Recognition Benchmarks | [TVT Repository](https://github.com/uta-smile/TVT) | [Paper](https://openaccess.thecvf.com/content/WACV2023/html/Yang_TVT_Transferable_Vision_Transformer_for_Unsupervised_Domain_Adaptation_WACV_2023_paper.html) |
| Office-31 | [FFTAT Repository](https://github.com/Shawey94/WACV2025-FFTAT) | [Paper](https://arxiv.org/pdf/2411.07794) |

## Pretrained ViT

Download the following ImageNet-21k pretrained backbones and place them in the pretrained_vit/ directory.

| Backbone | Download |
| --- | --- |
| ViT-B/16 | [Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) |
| ViT-L/16 | [Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz) |
| ViT-S/16 | [Download](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz) |

These weights are used for ViT backbone initialization during training.

## Pretrained Models

Download the pretrained model files from Baidu Netdisk:

- **Download:** [Baidu Netdisk](https://pan.baidu.com/s/1i8xIdziKM2r-8RGD_AESNw?pwd=1ecg)
- **Extraction Code:** `1ecg`

After downloading and extracting the files, place the model files
in the `checkpoint/` directory.

## Testing

### 1. Prepare the dataset

Download Oracle-241 using the dataset resources listed above and extract the images into the `data/` directory.

```text
CT-OBCR/
├── data/
├── checkpoint/
├── models/
├── inference_runtime.py
├── requirements.txt
└── test.py
```


### 2. Download the trained model

Download the trained CT-OBCR checkpoint from [Baidu Netdisk](https://pan.baidu.com/s/1i8xIdziKM2r-8RGD_AESNw?pwd=1ecg).

Extraction code: `1ecg`

After extraction, place `final.pt` at:

```text
checkpoint/final.pt
```

The ImageNet-21k weights listed under “Pretrained ViT” are used for training initialization and are not required for testing with `final.pt`.

### 3. Install dependencies

Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

For GPU testing, use a compatible CUDA-enabled PyTorch installation. The same installation can also execute on CPU.

### 4. Run evaluation

Run the following command from the repository root to evaluate on GPU:

```bash
python test.py --checkpoint checkpoint/final.pt --test-list data/scan_test.txt --data-root . --device cuda --output-dir results_gpu
```

For CPU evaluation:

```bash
python test.py --checkpoint checkpoint/final.pt --test-list data/scan_test.txt --data-root . --device cpu --output-dir results_cpu
```

The script displays cumulative accuracy during evaluation and saves the following files in the selected output directory:

- `accuracy.txt`: final accuracy, correct/total counts, and evaluation settings.
- `predictions.csv`: ground-truth labels, predicted classes, and confidence scores.

The evaluation batch size is read from the checkpoint. Preserve the supplied test-list order and saved batch size when comparing results, because predictions can depend on batch composition. Use a new output directory for each evaluation.

## Code and Model Availability

This repository provides the testing scripts and compiled core model modules required for inference and evaluation. The trained model checkpoint is available through the Baidu Netdisk link above.
## Acknowledgments

This work builds upon **TVT** and **FFTAT**. We sincerely thank the authors
for sharing their code, which provided valuable references for the
implementation of our framework.

- [TVT](https://github.com/uta-smile/TVT)
- [FFTAT](https://github.com/Shawey94/WACV2025-FFTAT)
