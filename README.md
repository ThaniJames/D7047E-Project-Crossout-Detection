## Cross-Out Detection and Classification in Handwritten Documents

## Project Overview

Handwriting recognition models struggle with crossed-out words in handwritten documents. This project builds and compares three progressively advanced deep learning models to detect and classify cross-outs, improving downstream OCR accuracy.

### Tasks
- **Task 1:** Binary classification — Clean vs Crossed-out
- **Task 2:** 7-class classification — identify the cross-out type (Single-Line, Double-Line, Diagonal, Cross, Wave, Zig-zag, Scratch)
- **Task 3:** Evaluate on 50 custom handwritten images to test generalization

### Models

| Model | Architecture | Status | Task 1 Acc | Task 2 Acc |
|-------|-------------|--------|-----------|-----------|
| Model 1 | EfficientNet-B0 (baseline) | Planned | - | - |
| Model 2 | EfficientNet-B0 + MTL | Planned | — | — |
| Model 3 | CNN + Transformer + MTL | Planned | — | — |

---

## Project Structure

```
crossout-detection/
├── notebooks/
│   ├── Model1_EfficientNet.ipynb         # EfficientNet-B0 baseline
│   ├── Model2_EfficientNet_MTL.ipynb     # EfficientNet + Multi-Task Learning
│   └── Model3_CNN_Transformer_MTL.ipynb  # CNN + Transformer + MTL
├── data/
│   ├── raw/                              # IAM dataset + synthetic cross-outs (4.3 GB)
│   │   ├── train/images/                 # 9 folders: CLEAN + 7 types + MIXED
│   │   ├── val/images/
│   │   └── test/images/
│   └── custom_dataset/                   # Task 3: ~50 handwritten images
├── models/                               # Saved weights (.pth) — not tracked in git
│   ├── model1_efficientnet/
│   ├── model2_cnn_transformer/
│   └── model3_cnn_transformer_mtl/
└── results/                              # Training curves, confusion matrices
    ├── model1/
    ├── model2/
    └── model3/
```

## Dataset

**Source:** IAM Handwriting Database with synthetic cross-outs (provided by course instructors)

| Split | Clean | Per Cross-out Type | Total |
|-------|------:|------------------:|------:|
| Train | 47,997 | 47,997 each | 431,973 |
| Val | 7,559 | 7,559 each | 68,031 |
| Test | 20,306 | 20,306 each | 182,754 |

Each split contains 9 image folders:
- `CLEAN/` — original word images (no cross-out)
- `SINGLE_LINE/`, `DOUBLE_LINE/`, `DIAGONAL/`, `CROSS/`, `WAVE/`, `ZIG_ZAG/`, `SCRATCH/` — cross-out types
- `MIXED/` — randomly assigned cross-out type per image

**Binary task** uses all 9 folders (CLEAN=0, rest=1). **Multiclass task** uses only the 7 type folders (MIXED excluded — no per-image type label).

---

## Model 1: EfficientNet-B0 Baseline (Done)

### Architecture
- EfficientNet-B0 pretrained on ImageNet, classifier head replaced for our tasks
- 4,008,829 parameters (all trainable — full fine-tuning)
- Separate models trained for Task 1 (binary) and Task 2 (7-class)

### Training Configuration
| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-4, weight_decay=1e-4) |
| Scheduler | ReduceLROnPlateau (patience=3, factor=0.5) |
| Early Stopping | patience=5 |
| Batch Size | 64 |
| Mixed Precision | AMP (float16 on CUDA) |
| Augmentation | Resize 224, rotation ±10°, color jitter, translation 5% |
| Class Balancing | WeightedRandomSampler for binary task (1:8 imbalance) |

### Results

**Task 1 — Binary Classification (Clean vs Crossed-out):**

| Metric | Score |
|--------|-------|
| Accuracy | 97.98% |
| F1-Score | 98.85% |
| Precision | 98.70% |
| Recall | 98.99% |
| AUC-ROC | 99.49% |

**Task 2 — 7-Class Cross-out Type Classification:**

| Metric | Score |
|--------|-------|
| Accuracy | 90.89% |
| Macro F1 | 91.36% |
| Macro Precision | 92.97% |

Per-class performance:

| Class | Precision | Recall | F1 |
|-------|-----------|--------|-----|
| Single-Line | 0.89 | 0.91 | 0.90 |
| Double-Line | 1.00 | 0.90 | 0.94 |
| Diagonal | 0.67 | 0.98 | 0.80 |
| Cross | 1.00 | 0.90 | 0.95 |
| Wave | 0.99 | 0.89 | 0.93 |
| Zig-zag | 0.98 | 0.89 | 0.93 |
| Scratch | 0.98 | 0.90 | 0.94 |

**Key Finding:** Diagonal class has low precision (0.67) — the CNN confuses local straight-stroke features at different angles. This is the primary motivation for adding Transformer attention in Model 3.

### Saved Artifacts
- `backbone_task2.pth` — backbone weights for Model 2/3 (progressive transfer learning)
- `task1_final.pth`, `task2_final.pth` — full model checkpoints with metadata

---

## Progressive Training Strategy

```
Model 1 (EfficientNet) → backbone_task2.pth
    ↓ load backbone weights
Model 2 (EfficientNet + MTL) → backbone.pth
    ↓ load backbone weights
Model 3 (CNN + Transformer + MTL)
```

Each model loads the previous model's backbone, adding its own improvements on top.

---

## Running

### Quick Test (1-2 minutes)
```python
TEST_MODE = True    # Uses only 200 images per split
NUM_WORKERS = 0     # On macOS
```

### Full Training (CUDA server)
```python
TEST_MODE = False
NUM_WORKERS = 4     # On Linux/CUDA
```

### Environment
- Python 3.x
- PyTorch 2.x with CUDA
- torchvision, scikit-learn, tqdm, Pillow, matplotlib
