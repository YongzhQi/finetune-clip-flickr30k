# finetune-clip-flickr30k

CPU-friendly utilities for fine-tuning OpenAI CLIP (ViT-B/32) on the Flickr30k
image-caption dataset. The project provides both a command-line script and a
Jupyter notebook that train only CLIP's projection heads (image/text
projections + temperature) and report retrieval metrics before and after
fine-tuning.

## Features
- Torchvision-backed data loader for Flickr30k (`captions.txt` + optional
  `train/val/test` split files).
- TSV fallback loader for custom manifests.
- Baseline vs. fine-tuned Recall@1/5/10 + Median Rank comparisons saved to JSON.
- Standalone evaluation helper (`scripts/compare_models.py`) to benchmark
  arbitrary checkpoints.
- Reproducible notebook (`notebooks/clip_finetuning_cpu.ipynb`) that mirrors the
  CLI workflow.

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected directory layout for the torchvision dataset:
```
/data/flickr30k/
├── images/                # flickr30k_images, original JPGs
├── annotations/
│   └── captions.txt       # official captions file
└── splits/                # optional lists: train.txt, val.txt, test.txt
```
If you rely on a TSV manifest instead, each line should be:
```
image_path<TAB>caption<TAB>image_id<TAB>split
```

## Quick Start (command line)
```bash
python finetune_clip_flickr30k_cpu.py \
  --torchvision-root /data/flickr30k/images \
  --torchvision-ann /data/flickr30k/annotations/captions.txt \
  --torchvision-split /data/flickr30k/splits \
  --epochs 1 --batch-size 32 --subset 5000
```

Zero-shot only (baseline metrics):
```bash
python finetune_clip_flickr30k_cpu.py --zero-shot \
  --torchvision-root /data/flickr30k/images \
  --torchvision-ann /data/flickr30k/annotations/captions.txt
```

Outputs include:
- `artifacts/clip_flickr30k_results.json` — baseline vs. tuned metrics.
- `checkpoints/linearprobe_best.pt` — best projection-head checkpoint.

## Notebook
Open `notebooks/clip_finetuning_cpu.ipynb` (locally or in Colab), edit the
`Config` cell with your paths, and run all cells.

## Model Comparison Utility
Use `scripts/compare_models.py` to evaluate two checkpoints side-by-side on a
held-out split (annotations provided via TSV).
```bash
python scripts/compare_models.py \
  --annotations data/flickr30k/test.tsv \
  --image-root /data/flickr30k/images \
  --finetuned-checkpoint checkpoints/linearprobe_best.pt
```

## License
Add your chosen license here (MIT, Apache-2.0, etc.).
