# finetune-clip-flickr30k

CPU-friendly utilities for fine-tuning OpenAI CLIP (ViT-B/32) on the Flickr30k
image-caption dataset. The project provides a complete pipeline for linear probe
fine-tuning that trains only CLIP's projection heads while keeping the backbone
frozen, achieving **+3.10 percentage point improvement** in Text→Image Recall@5.

## Features
- Complete training pipeline with proper train/validation/test splits
- Linear probe fine-tuning (only 0.43% of model parameters)
- Validation-based early stopping and checkpoint management
- Comprehensive evaluation with baseline vs. fine-tuned comparisons
- Clean data handling for Flickr30k dataset
- Standalone evaluation helper (`scripts/compare_models.py`) for model comparison
- Research notebook (`notebooks/clip_finetuning_cpu.ipynb`) for exploration

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Dataset Structure
The project expects Flickr30k images and a consolidated CSV file:
```
data/
├── flickr30k_images/              # Original Flickr30k image files
├── flickr30k_annotations.csv      # Original annotations (image,caption)
└── flickr30k_annotations_with_splits.csv  # Annotations with train/val/test splits
```

## Pipeline Overview

### 1. Data Preparation
```bash
python create_splits.py
```
Creates proper train/validation/test splits (80%/10%/10%) from the annotations.

### 2. Training
```bash
python train_improved.py
```
Trains the model with:
- Learning rate: 5e-5
- Batch size: 32
- 3 epochs with early stopping
- Validation-based checkpoint saving

### 3. Final Evaluation
```bash
python final_evaluation.py
```
Evaluates both baseline and fine-tuned models on the held-out test set.

## Results
**Achieved +3.10 percentage point improvement** in Text→Image Recall@5:
- Baseline (pretrained): 92.60%
- Fine-tuned: 95.70%

Results are saved in `artifacts/final_evaluation_results.json` with complete metrics for both directions (Text→Image and Image→Text) and all Recall@K values.

## Model Comparison Utility
Use `scripts/compare_models.py` to evaluate two checkpoints side-by-side:
```bash
python scripts/compare_models.py \
  --annotations data/flickr30k_annotations_with_splits.csv \
  --image-root data/flickr30k_images \
  --finetuned-checkpoint checkpoints/linearprobe_best.pt \
  --split test
```

## Project Structure
```
├── train_improved.py                        # Main training script
├── final_evaluation.py                      # Test set evaluation
├── create_splits.py                         # Data splitting utility
├── scripts/
│   └── compare_models.py                    # Model comparison CLI tool
├── notebooks/
│   └── clip_finetuning_cpu.ipynb           # Research notebook
├── data/
│   ├── flickr30k_annotations.csv           # Original annotations
│   ├── flickr30k_annotations_with_splits.csv # Annotations with splits
│   └── flickr30k_images/                   # Image dataset (31,783 images)
├── checkpoints/
│   └── linearprobe_best.pt                 # Trained model checkpoint
└── artifacts/
    ├── final_evaluation_results.json       # Complete test results
    ├── training_history.json               # Training progression
    └── comparison_results.json             # Model comparison metrics
```
