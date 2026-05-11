<p align="center">
  <img src="docs/gif/reaction.gif" alt="Reaction Demo" width="45%">
  <img src="docs/gif/trex.gif" alt="Trex Demo" width="45%">
</p>

# [CVPR 2026 Highlight] Object-DINO


[![Paper](https://img.shields.io/badge/Paper-arXiv%3A2603.26127-b31b1b.svg)](https://arxiv.org/abs/2603.26127)
[![Project Page](https://img.shields.io/badge/Project-blue)](https://samyakr99.github.io/Object_dino/)
[![CVPR 2026 (Highlight)](https://img.shields.io/badge/CVPR%202026-Highlight-red)](https://cvpr.thecvf.com/)

**Object-DINO** is a training-free method that extracts distributed, object-centric information from self-supervised Vision Transformers (such as DINO). It leverages this localized visual evidence for two applications: unsupervised object discovery and mitigating object hallucinations in Multimodal Large Language Models (MLLMs).

---

## Repository Structure

```
Object-DINO/
├── Demo.ipynb     
├── unsupervised_object_discovery/     
└── mllm_hallucinations/               
    ├── coco/                          
    ├── chair/                         
    ├── pope/                          
    └── mme/                           
```

---

## Environments

| Environment | Used For |
|---|---|
| `dinov3_env` | Unsupervised Object Discovery |
| `llava` | MLLM guidance generation  |
| `marine` | POPE & CHAIR evaluation |

### Creating the Environments

```bash
# Create the environments
conda env create -f envs/dinov3_env.yml
conda env create -f envs/llava.yml
conda env create -f envs/marine.yml
```

---

## Application 1: Unsupervised Object Discovery

This application replaces the standard TokenCut baseline (which natively uses all final-layer heads) with our custom, dynamically selected set of object-centric heads distributed across the network (`object_dino_feature_extraction.py`).

### Datasets
Please refer to [`Download_data.md`](Download_data.md) for downloading VOC2007, VOC2012, and COCO 2014 datasets.

### Full Run (VOC07 + VOC12 + COCO20k)

```bash
sbatch unsupervised_object_discovery/run_object_discovery.sh
```

### Key Arguments

| Argument | Description | Value Used |
|---|---|---|
| `--dataset` | Dataset name | `VOC07`, `VOC12`, `COCO20k` |
| `--set` | Split | `trainval` / `train` |
| `--which_features` | Feature type | `object_dino` (our method) |
| `--arch` | Backbone | `vit_base` |
| `--tau` | Graph threshold | `-0.35` |

---

## Application 2: MLLM Hallucination Mitigation

Our method provides explicit visual grounding by generating object-centric similarity maps using DINOv3. These maps are used to guide LLaVA 1.5's decoding process via logit blending, amplifying tokens that are consistent with the visual evidence to reduce hallucination:

```
combined_logits = α · logits(original image) + (1 - α) · logits(highlighted image)
```

### Highlighted Images

To generate highlighted images, run `dino_coco.py`, `dino_pope.py`, or `dino_mme.py` using the `dinov3_env` environment.

---

### Step 1 — Guidance Generation (`llava` env)

Runs LLaVA 1.5 with α-guided decoding across all three benchmarks:

```bash
sbatch mllm_hallucinations/run_quick_test.sh
```

This runs sequentially:
1. `pope/guidance_pope.py` — answers POPE yes/no questions with guidance
2. `coco/guidance_coco.py` — generates guided captions for CHAIR evaluation
3. `mme/guidance_mme.py` — generates guided answers for MME tasks

---

### Step 2 — Evaluation (`marine` env)

```bash
conda activate marine
bash mllm_hallucinations/eval_quick_test.sh
```

This runs:
1. **POPE**: `convert_pope.py` → `eval_pope.py`
2. **CHAIR**: `chair_alpha.sh` (converts JSON → runs `chair.py`)
3. **MME**: `eval_mme.py`

---

### Running Individual Datasets

#### POPE only
```bash
# Guidance (llava env)
cd mllm_hallucinations/pope
CUDA_VISIBLE_DEVICES=0 python -u guidance_pope.py

# Eval (marine env)
python convert_pope.py
python eval_pope.py
```

#### CHAIR only
```bash
# Guidance (llava env)
cd mllm_hallucinations/coco
CUDA_VISIBLE_DEVICES=0 python -u guidance_coco.py

# Eval (marine env)
cd mllm_hallucinations/chair
bash chair_alpha.sh
```

#### MME only
```bash
# Guidance (llava env)
cd mllm_hallucinations/mme
CUDA_VISIBLE_DEVICES=0 python -u guidance_mme.py

# Eval
python eval_mme.py
```

---

## Acknowledgments

This repository builds upon several excellent open-source projects. We would like to thank the authors of:
- [TokenCut](https://github.com/YangtaoWANG95/TokenCut)
- [DeGF](https://github.com/zhangce01/DeGF)

---

## Citation

If you find this work useful, please consider citing:

```bibtex
@article{rawlekar2026finding,
  title={Finding Distributed Object-Centric Properties in Self-Supervised Transformers},
  author={Rawlekar, Samyak and Swain, Amitabh and Cai, Yujun and Wang, Yiwei and Yang, Ming-Hsuan and Ahuja, Narendra},
  journal={arXiv preprint arXiv:2603.26127},
  year={2026}
}
```
