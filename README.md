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

Download the following ImageNet-21k pretrained backbones and place them
in the `checkpoint/` directory. Save the files using the filenames listed below.

| Backbone | Download | Local Filename |
| --- | --- | --- |
| ViT-B/16 | [Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz) | `imagenet21k_ViT-B_16.npz` |
| ViT-L/16 | [Download](https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz) | `imagenet21k_ViT-L_16.npz` |
| ViT-S/16 (AugReg) | [Download](https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz) | `S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz` |

These checkpoints initialize the ViT backbones. Trained CT-OBCR model
checkpoints will be released separately.

## Code and Model Availability

The source code and trained model weights will be publicly released
in this repository.

Installation instructions, model download links, and evaluation commands
will be provided with the release.

## Acknowledgments

This work builds upon **TVT** and **FFTAT**. We sincerely thank the authors
for sharing their code, which provided valuable references for the
implementation of our framework.

- [TVT](https://github.com/uta-smile/TVT)
- [FFTAT](https://github.com/Shawey94/WACV2025-FFTAT)
