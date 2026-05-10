# Object-DINO 🦕

**Object-DINO** leverages DINOv3 features for two applications: unsupervised object discovery and reducing hallucinations in Multi-modal Large Language Models (MLLMs).

---

## 📂 Repository Structure

```
Object-DINO/
├── analysis/                          # Analysis and evaluation scripts
├── unsupervised_object_discovery/     # Application 1: Object discovery via TokenCut
└── mllm_hallucinations/               # Application 2: MLLM hallucination mitigation
    ├── coco/                          # COCO guided caption generation
    ├── chair/                         # CHAIR evaluation
    ├── pope/                          # POPE evaluation
    └── mme/                           # MME evaluation
```

---

## ⚙️ Environments

| Environment | Used For |
|---|---|
| `dinov3_env` | Unsupervised Object Discovery |
| `llava` | MLLM guidance generation (LLaVA 1.5) |
| `marine` | POPE & CHAIR evaluation (login node, no GPU) |

### Creating the Environments

We provide exported conda environments in the `envs/` directory. You can recreate them using:

```bash
# Create the environments
conda env create -f envs/dinov3_env.yml
conda env create -f envs/llava.yml
conda env create -f envs/marine.yml
```

---

## 🔍 Application 1: Unsupervised Object Discovery

Runs TokenCut with our custom DINOv3-based feature extraction (`object_dino_feature_extraction.py`) on standard object discovery benchmarks.

### Datasets
Datasets are pre-stored at `/scratch/bcyh/dataset/` — no downloads needed.
- `VOC07` → `/scratch/bcyh/dataset/VOC2007/`
- `VOC12` → `/scratch/bcyh/dataset/VOC2012/`
- `COCO20k` → `/scratch/bcyh/dataset/coco20k/`

### Quick Test (VOC07)

```bash
# Interactive (srun)
srun --account=bcyh-delta-gpu --partition=gpuA40x4 --gpus=1 --mem=32g --time=00:30:00 --pty bash -c "
source /scratch/bcyh/miniconda3/etc/profile.d/conda.sh && conda activate dinov3_env &&
cd /scratch/bcyh/samyakr99/Object-DINO/unsupervised_object_discovery &&
python -u main_tokencut_copy.py --dataset VOC07 --set trainval --which_features object_dino --arch vit_base --tau -0.35
"

# Or as a batch job
sbatch unsupervised_object_discovery/run_quick_test.sh
```

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

## 🧠 Application 2: MLLM Hallucination Mitigation

Our method generates attention-highlighted images using DINOv3, then uses them to guide LLaVA 1.5's decoding via logit blending:

```
combined_logits = α · logits(original image) + (1 - α) · logits(highlighted image)
```

### Highlighted Images
Pre-generated highlighted images are stored at:
- **COCO/CHAIR**: `/scratch/bcyh/samyakr99/chair_experiment/results/` (501 images)
- **POPE**: `/scratch/bcyh/samyakr99/chair_experiment/results_pope_*/` (500 each: adv, popular, random)
- **MME**: `/scratch/bcyh/samyakr99/chair_experiment/results_mme/` (existence, color, count, position)

> To regenerate highlighted images, run `dino_coco.py` / `dino_pope.py` / `dino_mme.py` with `dinov3_env`.

---

### Step 1 — Guidance Generation (GPU, `llava` env)

Runs LLaVA 1.5 with α-guided decoding across all three benchmarks:

```bash
sbatch mllm_hallucinations/run_quick_test.sh
```

Monitor logs:
```bash
tail -f mllm_hallucinations/quick_test_<JOBID>.log
```

This runs sequentially:
1. `pope/guidance_pope.py` — answers POPE yes/no questions with guidance
2. `coco/guidance_coco.py` — generates guided captions for CHAIR evaluation
3. `mme/guidance_mme.py` — generates guided answers for MME tasks

---

### Step 2 — Evaluation (Login Node, `marine` env)

Run **after** Step 1 completes. No GPU needed.

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
# Guidance (llava env, GPU)
cd mllm_hallucinations/pope
CUDA_VISIBLE_DEVICES=0 python -u guidance_pope.py

# Eval (marine env, login node)
python convert_pope.py
python eval_pope.py
```

#### CHAIR only
```bash
# Guidance (llava env, GPU)
cd mllm_hallucinations/coco
CUDA_VISIBLE_DEVICES=0 python -u guidance_coco.py

# Eval (marine env, login node)
cd mllm_hallucinations/chair
bash chair_alpha.sh
```

#### MME only
```bash
# Guidance (llava env, GPU)
cd mllm_hallucinations/mme
CUDA_VISIBLE_DEVICES=0 python -u guidance_mme.py

# Eval
python eval_mme.py
```

---

## 📖 Citation

If you find this work useful, please consider citing:

```bibtex
@article{object_dino2024,
  title     = {Object-DINO: Grounding MLLMs and Unsupervised Discovery},
  author    = {},
  journal   = {arXiv},
  year      = {2024}
}
```
