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

## Benchmarks

The framework is evaluated on:
- Oracle-241.
- Three digit recognition adaptation tasks.
- Six Office-31 adaptation tasks.

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
