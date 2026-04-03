# ExCeL: OpenOOD-Based Reproduction and Evaluation

This repository provides our implementation pipeline for **ExCeL** under the **OpenOOD** benchmark setting.

We follow the **OpenOOD benchmark protocol**, dataset splits, and directory organization:
- [OpenOOD GitHub](https://github.com/Jingkang50/OpenOOD)
- [OpenOOD Benchmark Overview](https://github.com/Jingkang50/OpenOOD/wiki/OpenOOD-v1.5-methods-%26-benchmarks-overview)

---

## Contents
- [Overview](#overview)
- [Summary of Results](#summary-of-results)
- [Data Download and Preparation](#data-download-and-preparation)
  - [Download](#download)
  - [Preparation](#preparation)
  - [Expected File Structure](#expected-file-structure)
- [Pretrained Model](#pretrained-model)
- [Evaluation](#evaluation)
- [Reference](#reference)
- [Acknowledgement](#acknowledgement)

## Overview

This repository is built for reproducing and evaluating **ExCeL** on the **OpenOOD benchmark**.

**ExCeL** (Combined **Ex**treme and **C**oll**e**ctive **L**ogit Information) is a post-hoc OOD detector that combines extreme information (maximum logit of the top predicted class) with collective information (probability of other classes appearing in subsequent ranks across training samples) for enhanced and consistent OOD detection performance.

Currently supported settings include:
- **CIFAR-100** with **ResNet-18 (32x32)**
- **ImageNet-200** with **ResNet-18 (224x224)**

## Summary of Results

| Dataset | Backbone | Near-OOD AUROC | Near-OOD FPR@95 | Far-OOD AUROC | Far-OOD FPR@95 |
| --- | --- | --- | --- | --- | --- |
| CIFAR-100 | ResNet-18 (32x32) | 80.64±0.02 | 55.99±0.10 | 82.15±0.08 | 52.50±0.29 |
| ImageNet-200 | ResNet-18 (224x224) | 82.38±0.04 | 58.29±0.21 | 92.09±0.02 | 28.62±0.05 |

## Data Download and Preparation

This repository follows the OpenOOD data organization. By default, the code reads:

- benchmark image lists from `./data/benchmark_imglist/`
- small-scale image datasets from `./data/images_classic/`
- large-scale image datasets from `./data/images_largescale/`
- experiment outputs from `./results/`

### Download

If you would like to also use OpenOOD for training, you can get most benchmark data with the [OpenOOD downloading script](https://github.com/Jingkang50/OpenOOD/tree/main/scripts/download).

- For **CIFAR-100**, use OpenOOD download/preparation scripts.
- For **ImageNet-1K**, training images should be downloaded from the [ImageNet Official Website](https://www.image-net.org/download.php).
- After downloading, place files in the OpenOOD-style directory structure below.

### Preparation

Recommended workflow:

1. Download the datasets required by your experiment.
2. Place image files under `data/images_classic/` or `data/images_largescale/`.
3. Make sure the corresponding image-list files exist under `data/benchmark_imglist/`.
4. Place pretrained checkpoints under `results/...` and update checkpoint paths in configs/scripts if needed.

Dataset config files used by ExCeL scripts:

- CIFAR-100 ID data: `configs/datasets/cifar100/cifar100.yml`
- CIFAR-100 OOD splits: `configs/datasets/cifar100/cifar100_ood.yml`
- ImageNet-200 ID data: `configs/datasets/imagenet200/imagenet200.yml`
- ImageNet-200 OOD splits: `configs/datasets/imagenet200/imagenet200_ood.yml`

### Expected File Structure

```text
├── data
│   ├── benchmark_imglist
│   ├── images_classic
│   └── images_largescale
├── configs
├── openood
├── scripts
├── main.py
└── README.md
```

## Pretrained Model

Pre-trained models and checkpoints can be downloaded from:
- CIFAR-100 [[Google Drive]](https://drive.google.com/file/d/1s-1oNrRtmA0pGefxXJOUVRYpaoAML0C-/view?usp=drive_link): ResNet-18 classifiers trained with cross-entropy loss from 3 training runs.
- ImageNet-200 [[Google Drive]](https://drive.google.com/file/d/1ddVmwc8zmzSjdLUO84EuV4Gz1c7vhIAs/view?usp=drive_link): ResNet-18 classifiers trained with cross-entropy loss from 3 training runs.

## Evaluation

Please ensure dataset paths and checkpoint paths are correctly set before evaluation.

### CIFAR-100

```bash
python main.py \
    --config configs/datasets/cifar100/cifar100.yml \
    configs/datasets/cifar100/cifar100_ood.yml \
    configs/networks/resnet18_32x32.yml \
    configs/pipelines/test/test_excel.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/excel.yml \
    --network.checkpoint 'results/cifar100_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt' \
    --mark 1
```

### ImageNet-200

```bash
python main.py \
    --config configs/datasets/imagenet200/imagenet200.yml \
    configs/datasets/imagenet200/imagenet200_ood.yml \
    configs/networks/resnet18_224x224.yml \
    configs/pipelines/test/test_excel.yml \
    configs/preprocessors/base_preprocessor.yml \
    configs/postprocessors/excel.yml \
    --network.checkpoint 'results/imagenet200_resnet18_224x224_base_e90_lr0.1_default/s0/best.ckpt' \
    --mark 1
```


## Reference

This project follows the OpenOOD benchmark protocol, dataset splits, and directory organization.

If you find ExCeL useful, please consider citing:

```bibtex
@article{karunanayake2023excel,
  title={ExCeL: Combined Extreme and Collective Logit Information for Enhancing Out-of-Distribution Detection},
  author={Karunanayake, Naveen and Seneviratne, Suranga and Chawla, Sanjay},
  journal={arXiv preprint arXiv:2311.14754},
  year={2023}
}
```

If you use this benchmark setup, please also cite OpenOOD:

```bibtex
@article{zhang2023openood,
  title={OpenOOD v1.5: Enhanced Benchmark for Out-of-Distribution Detection},
  author={Zhang, Jingyang and Yang, Jingkang and Wang, Pengyun and Wang, Haoqi and Lin, Yueqian and Zhang, Haoran and Sun, Yiyou and Du, Xuefeng and Li, Yixuan and Liu, Ziwei and Chen, Yiran and Li, Hai},
  journal={arXiv preprint arXiv:2306.09301},
  year={2023}
}
```


## Acknowledgement

This repository is built on top of the [OpenOOD benchmark](https://github.com/Jingkang50/OpenOOD) and follows its benchmark protocol and data organization.
