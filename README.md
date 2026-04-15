# CC-DiT

CC-DiT is a conditional cold diffusion framework for retinal vessel segmentation.
This repository provides the training and inference code for structure-preserving retinal vessel segmentation with conditional restoration, attention-based feature refinement, and topology-aware supervision.

---

## Overview

Retinal vessel segmentation is challenging because of thin vessels, low contrast, vessel crossings, and lesion-like interference.
CC-DiT formulates segmentation as a conditional cold diffusion restoration process, where a vessel mask is reconstructed progressively under the guidance of the corresponding fundus image.

The current implementation includes:

- conditional cold diffusion segmentation
- CBAM-based feature refinement
- ViT-based global context modeling
- hybrid loss with Dice, clDice, and BCE
- sliding-window inference for high-resolution retinal images

---

## Framework

> Replace the image path below with your actual framework figure if needed.

<p align="center">
  <img src="figures/model.pdf" alt="CC-DiT Framework" width="900">
</p>


---

## Repository Structure

```text
CC-DiT/
├── .vscode/
├── figures/
├── gycutils/
├── model/
│   ├── SegDiffusion.py
│   ├── loss.py
│   ├── datasets.py
│   ├── cldice.py
│   ├── soft_skeleton.py
│   └── ...
├── data_augmentation.py
├── train.py
├── sample.py
├── requirements.txt
└── README.md