# ViT-Tiny Based Colorectal Histopathology Image Classification

This project implements a Vision Transformer Tiny (ViT-Tiny) deep learning pipeline for multiclass colorectal histopathology image classification using a merged dataset constructed from NCT-CRC-HE-100K and EBHI-HE image collections.

The objective is to classify colorectal tissue and lesion patterns into 13 diagnostic categories and evaluate transformer-based performance on histopathological image analysis.

---

## Dataset Used

### 1. NCT-CRC-HE-100K
Public colorectal cancer histology patch dataset containing tissue-level classes.

### 2. EBHI-HE (200x Magnification)
Endoscopic biopsy histopathological image dataset containing lesion-level disease classes.

### Final Merged Classes (13)

- ADI
- Adenocarcinoma
- BACK
- DEB
- High-grade IN
- LYM
- Low-grade IN
- MUC
- MUS
- NORM
- Polyp
- STR
- TUM

Dataset preprocessing included:

- merging heterogeneous sources
- class balancing
- 80/10/10 train-validation-test split
- image normalization and augmentation
- class weighted loss handling

---

## Model Architecture

- Backbone: ViT-Tiny Patch16 224
- Pretrained Weights: ImageNet
- Framework: PyTorch + timm
- Loss Function: Weighted CrossEntropyLoss
- Optimizer: AdamW
- Device: CPU Training

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 8 |
| Epochs | 5 |
| Learning Rate | 1e-4 |
| Input Size | 224 x 224 |
| Number of Classes | 13 |

---

## Final Performance

### Best Validation Accuracy
97.65%

### Final Test Accuracy
97.85%

### Macro Average F1-Score
94.92%

### Weighted Average F1-Score
97.86%

---

## Generated Outputs

- accuracy_curve.png
- loss_curve.png
- confusion_matrix.png
- vit_test_predictions.csv
- best_vit_tiny.pth

---

## Project Files

- `balanced_split.py` → balanced dataset creation and split
- `train_vit_final.py` → ViT-Tiny training pipeline
- `test_vit_final.py` → final unseen test evaluation
- `results_visualization.py` → graph and confusion matrix generation

---

## How to Run

### 1. Install Requirements

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
#### Place merged dataset inside:
```bash
NCT-CRC-HE-100K/
```
#### Run
```bash
python balanced_split.py
```

### 3. Train Model
```bash
python train_vit_final.py
```

### 4. Final test Evaluation
```bash
python test_vit_final.py
```

### 5. Generate Result Visualizations
```bash
python results_visualization.py
```

## Technologies Used

-Python
-PyTorch
-timm
-torchvision
-scikit-learn
-matplotlib
-pandas