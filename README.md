# Diagnosing Adaptation Failure Modes in Foundation Models for RNFL Thickness Prediction

[![Conference](https://img.shields.io/badge/MICCAI%20OMIA%202026-Oral%20Paper-blue.svg)](https://github.com/aryan355-pr/radial-aggregator-rnfl)
[![Python](https://img.shields.io/badge/Python-3.10%2B-brightgreen.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange.svg)](https://pytorch.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

**TL;DR:** Standard classification-centric adaptation protocols (progressive layer freezing, shape-preserving Pearson losses) paradoxically degrade continuous dense metric regression by **18–32%** on en face IR-SLO fundus imaging. We identify two distinct failure modes: a saturating freezing threshold that collapses predictive variance and template overfitting under correlation penalties. Applying a first-order gradient penalty probe restores patient-specific prediction variance ($\sigma_{\text{pred}} \approx 11.8\,\mu\text{m}$) without compromising global accuracy ($19.04 \pm 0.03\,\mu\text{m}$).

---

## 🎯 Quick Start

### 1. Install Dependencies
```bash
git clone [https://github.com/aryan355-pr/radial-aggregator-rnfl.git](https://github.com/aryan355-pr/radial-aggregator-rnfl.git)
cd radial-aggregator-rnfl
pip install -r requirements.txt
```

### 2. Download Pre-trained Checkpoints
```bash
# Download best diagnostic checkpoint (Gradient-Loss probe)
wget [CHECKPOINT_LINK]/gradient_loss_best.pth -P checkpoints/
```

### 3. Run Inference on GRAPE (Cross-Modality Benchmark)
```bash
python scripts/evaluate.py \
    --config configs/gradient_loss.yaml \
    --checkpoint checkpoints/gradient_loss_best.pth \
    --data_path ./data/GRAPE \
    --output results/grape_predictions.csv
```
**Expected Output:** Global MAE $\approx 19.88\,\mu\text{m}$, $\sigma_{\text{pred}} \approx 12.15\,\mu\text{m}$ (Preserved anatomical variance).

---

## 📊 Main Results

### Table 1: Multi-Backbone Performance and Metric Dissociation (FairFedMed IID Split)

| Backbone | Pre-training Objective | Protocol | MAE (μm) ↓ | Pearson R ↑ | $\sigma_{\text{pred}}$ (μm) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| ImageNet (ViT-B/16) | Supervised Classification | Naive | 19.79 | 0.661 | 12.4 |
| | | Aggressive | 19.92 | 0.661 | 12.2 |
| | | Structured | 24.68 | 0.649 | 10.8 |
| DepthAnything (V2-Small) | Relative/Metric Depth | Naive | 19.93 | 0.661 | 12.5 |
| | | Aggressive | 19.88 | 0.662 | 12.3 |
| | | Structured | 24.75 | 0.645 | 10.6 |
| RETFound (ViT-L/16) | Self-Supervised Retinal MAE | Naive | 19.52 | 0.662 | 12.3 |
| | | Aggressive | 19.25 | 0.676 | 12.3 |
| | | **Gradient-Loss (Probe)** | **19.04 ± 0.03** | **0.676** | **11.8 ± 0.2** |
| | | Structured | 23.71 | 0.513 | 4.8 (Collapse) |

*Ground-truth target population standard deviation: $\sigma_{\text{true}} = 18.6\,\mu\text{m}$.*

### Table 2: Impact of Progressive Layer Freezing (RETFound ViT-L/16, MAE-Only)

| Configuration | Frozen Fraction | MAE (Δ, μm) | Pearson R | $\sigma_{\text{pred}}$ (μm) | Failure Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Aggressive (Ref.) | 0% | 19.25 (—) | 0.676 | 12.3 | Optimal Adaptation |
| freeze_4 | 17% | 23.07 (+20%) | 0.523 | 12.0 | Representation Jump |
| freeze_8 | 33% | 23.57 (+22%) | 0.512 | 11.7 | Plateau Saturation |
| freeze_12 | 50% | 23.71 (+23%) | 0.513 | 4.8 | Variance Collapse |

*Note: Progressive layer freezing exhibits an immediate non-linear representation penalty (+3.82 $\mu\text{m}$ at 17% frozen) rather than a linear trajectory, concluding in catastrophic expressive variance collapse at 50% freezing.*

### Table 3: Dissociation of Metrics Under Shape-Preserving Losses (Pearson Penalty $\lambda_p$)

| Configuration | $\lambda_p$ | MAE (Δ, μm) | Pearson R | $\sigma_{\text{pred}}$ (μm) | Diagnosis |
| :--- | :--- | :--- | :--- | :--- | :--- |
| MAE-only (Ref.) | 0 | 19.76 (—) | 0.661 | 12.3 | Target Baseline |
| MAE + P(0.5) | 0.5 | 24.80 (+25%) | 0.458 | 8.1 | Initial Collapse |
| MAE + P(10) | 10 | 25.22 (+28%) | 0.649 | 6.4 | Shape Overfitting |
| MAE + P(30) | 30 | 25.81 (+31%) | 0.661 | 4.8 | Template Overfitting Trap |

### Table 4: Sector-Wise Cross-Modality Validation on GRAPE Dataset

| Protocol | Global MAE ↓ | Superior ↓ | Nasal ↓ | Inferior ↓ | Temporal ↓ | $\sigma_{\text{pred}}$ (μm) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| Structured (MAE-Only) | 20.12 | 20.45 | 18.90 | 22.34 | 18.79 | 0.82 (Collapsed) |
| **Gradient-Loss (Probe)** | **19.88** | **19.61** | **18.56** | **22.12** | **19.25** | **12.15 (Preserved)** |

---

## 🏗️ Method Pipeline

```text
Input IR-SLO Image (224×224) 
         │
         ▼
Vision Transformer Encoder (RETFound ViT-L/16)
         │ [Extracts 196 × 1024 patch tokens]
         ▼
Projection & Upsampling Head (3-layer MLP → 56×56 Depth Map)
         │
         ▼
Radial Aggregator (Differentiable Circular Sampling @ 30% disc radius)
         │ [Extracts 360 angular coordinate features]
         ▼
RNFL Prediction Head (MLP with LayerNorm & Dropout)
         │
         ▼
Continuous 360° RNFL Thickness Profile T ∈ ℝ³⁶⁰
```

### Diagnostic Objectives

1. **Unconstrained Base Loss:**

$$\mathcal{L}_{\text{MAE}} = \frac{1}{360} \sum_{\theta=1}^{360} |T_{\text{pred}}(\theta) - T_{\text{gt}}(\theta)|$$

2. **First-Order Gradient-Preservation Probe:**

$$\mathcal{L}_{\text{Grad}} = \|T_{\text{pred}} - T_{\text{gt}}\|_1 + \lambda_g \|\nabla_\theta T_{\text{pred}} - \nabla_\theta T_{\text{gt}}\|_1$$

where $\nabla_\theta T(\theta) = T(\theta) - T(\theta - 1)$ is evaluated circularly ($\theta = 0$ wraps to $359$).
---

## 📦 Datasets & Setup

Due to Data Use Agreements (DUAs) and privacy compliance, users must acquire the source datasets directly:

1. **FairFedMed (Primary Benchmark):**
   * 11,539 curated en face IR-SLO images with matched 360-point OCT-derived ground-truth profiles (70/15/15 split).
   * Access via the official [FairFedMed Repository](https://github.com/Harvard-AI-and-Robotics-Lab/FairFedMed).
2. **GRAPE (Cross-Modality Benchmark):**
   * 243 eyes evaluated for out-of-distribution robustness across sector profiles.
   * Access via the official [GRAPE Figshare Collection](https://springernature.figshare.com/collections/GRAPE_A_multimodal_glaucoma_dataset_of_follow-up_visual_field_and_fundus_images_for_glaucoma_management/6406319/1).

```bash
# 1. Preprocess FairFedMed (Applies Lanczos-4 anti-aliasing interpolation)
python scripts/prepare_fairfedmed.py \
    --input_dir /path/to/downloaded/fairfedmed \
    --output_dir ./data/FairFedMed

# 2. Preprocess GRAPE (Standardizes polar quadrant alignment)
python scripts/prepare_grape.py \
    --input_dir /path/to/downloaded/grape \
    --output_dir ./data/GRAPE
```

---

## 🔬 Reproducing Paper Experiments

```bash
# Run complete multi-backbone evaluation suite
bash scripts/reproduce_paper.sh

# Train single model configuration (e.g., Gradient-Loss Probe)
python scripts/train.py \
    --config configs/gradient_loss.yaml \
    --data_path ./data/FairFedMed \
    --output_dir checkpoints/gradient_loss

# Run progressive freezing ablation (Table 2)
python scripts/ablation.py \
    --study freezing \
    --data_path ./data/FairFedMed

# Run Pearson penalty complexity ablation (Table 3)
python scripts/ablation.py \
    --study loss_complexity \
    --data_path ./data/FairFedMed
```

---

## 💾 Pre-trained Checkpoints

| Model Configuration | Backbone | MAE (μm) | $\sigma_{\text{pred}}$ (μm) | Download Link |
| :--- | :--- | :--- | :--- | :--- |
| Gradient-Loss (Probe) | RETFound ViT-L/16 | 19.04 | 11.8 | [RELEASE_LINK] |
| Aggressive Baseline | RETFound ViT-L/16 | 19.25 | 12.3 | [RELEASE_LINK] |
| Structured Protocol | RETFound ViT-L/16 | 23.71 | 4.8 | [RELEASE_LINK] |

---

## ⚠️ Statement of Scope & Research Notice

* **Research Purpose Only:** This codebase and accompanying checkpoints are developed strictly for academic investigation and diagnostic benchmarking. They are not cleared or intended for clinical diagnostics or patient management.
* **Task Boundary:** Empirical claims in this repository are specifically bounded to 1D continuous circular manifold regression of RNFL thickness from en face IR-SLO imaging[cite: 1].

---

## 📚 Citation

```bibtex
@inproceedings{bijva2026diagnosing,
  title={Diagnosing Adaptation Failure Modes in Foundation Models for RNFL Thickness Prediction},
  author={Bijva, Aryan and Suthar, Krunal},
  booktitle={Ophthalmic Medical Image Analysis (OMIA), MICCAI Workshops},
  year={2026}
}
```
