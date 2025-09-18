"""Compare baseline and fine-tuned CLIP models on Flickr30k retrieval metrics.

This script loads both a baseline CLIP checkpoint (pretrained or user-supplied)
and a fine-tuned checkpoint, evaluates them on the specified Flickr30k split,
and prints/serializes Recall@K metrics for image-to-text and text-to-image
retrieval.

Dataset expectations
--------------------
`annotations` must be a CSV with at least the following columns:
    - image (path of image file relative to `image_root`)
    - caption (string)
    - split (e.g., train/val/test)
Each image should appear with all five captions. For accurate Recall@K, the CSV
must aggregate the complete Flickr30k test split.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image

try:
    import open_clip  # type: ignore
except ImportError as exc:  # pragma: no cover - import guard
    raise SystemExit("`open_clip_torch` is required. Install via pip or conda.") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CLIP models on Flickr30k")
    parser.add_argument("--annotations", type=Path, required=True, help="CSV with Flickr30k captions and splits")
    parser.add_argument("--image-root", type=Path, required=True, help="Directory containing Flickr30k images")
    parser.add_argument("--split", type=str, default="test", help="Split name to evaluate (default: test)")
    parser.add_argument("--model-name", type=str, default="ViT-B-32", help="open_clip model name")
    parser.add_argument("--pretrained", type=str, default="openai", help="open_clip pretrained tag")
    parser.add_argument("--baseline-checkpoint", type=Path, help="Optional path to a custom baseline state_dict")
    parser.add_argument("--finetuned-checkpoint", type=Path, required=True, help="Fine-tuned model state_dict (.pt)")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size for feature extraction")
    parser.add_argument("--num-workers", type=int, default=0, help="Dataloader workers (CPU-friendly default: 0)")
    parser.add_argument("--output-json", type=Path, help="Optional path to store comparison metrics as JSON")
    return parser.parse_args()


def load_model(
    model_name: str, pretrained: str, checkpoint: Path | None
) -> Tuple[torch.nn.Module, Callable, Callable]:
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained, device="cpu"
    )
    tokenizer = open_clip.get_tokenizer(model_name)

    if checkpoint is not None:
        state = torch.load(checkpoint, map_location="cpu")
        if "state_dict" in state:
            state = state["state_dict"]
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            print(f"[warn] Missing keys when loading {checkpoint}: {missing}")
        if unexpected:
            print(f"[warn] Unexpected keys when loading {checkpoint}: {unexpected}")

    model.eval()
    return model, preprocess, tokenizer


def _batched(iterable: Iterable, batch_size: int) -> Iterable[List]:
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def encode_images(model: torch.nn.Module, preprocess, image_paths: List[Path], batch_size: int) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for batch_paths in _batched(image_paths, batch_size):
            images = []
            for path in batch_paths:
                with Image.open(path) as img:
                    image = img.convert("RGB")
                images.append(preprocess(image))
            image_batch = torch.stack(images)
            feat = model.encode_image(image_batch)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)


def encode_texts(model: torch.nn.Module, tokenizer, captions: List[str], batch_size: int) -> torch.Tensor:
    features = []
    with torch.no_grad():
        for batch_captions in _batched(captions, batch_size):
            tokens = tokenizer(batch_captions)
            feat = model.encode_text(tokens)
            features.append(feat.cpu())
    return torch.cat(features, dim=0)


def normalize_features(feats: torch.Tensor) -> torch.Tensor:
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def compute_recall_at_k(similarity: np.ndarray, positives: List[List[int]], ks: Iterable[int]) -> Dict[str, float]:
    if not positives:
        return {f"R@{k}": float("nan") for k in ks}

    recalls: Dict[str, float] = {}
    for k in ks:
        hits = 0
        for row_idx, positive_indices in enumerate(positives):
            k_eff = min(k, similarity.shape[1])
            if k_eff == 0:
                continue
            topk = np.argpartition(-similarity[row_idx], k_eff - 1)[:k_eff]
            if any(idx in topk for idx in positive_indices):
                hits += 1
        recalls[f"R@{k}"] = hits / len(positives)
    return recalls


def evaluate(
    model: torch.nn.Module,
    preprocess,
    tokenizer,
    df: pd.DataFrame,
    image_root: Path,
    batch_size: int,
    split: str | None,
) -> Dict[str, Dict[str, float]]:
    if split and "split" in df.columns:
        split_df = df[df["split"] == split]
        if split_df.empty:
            raise ValueError(f"No rows found for split='{split}'")
    else:
        split_df = df

    unique_images = split_df["image"].unique().tolist()
    image_idx = {img: idx for idx, img in enumerate(unique_images)}

    image_paths = [image_root / img for img in unique_images]
    captions = split_df["caption"].tolist()
    caption_image_indices = split_df["image"].map(image_idx).tolist()

    image_features = normalize_features(
        encode_images(model, preprocess, image_paths, batch_size)
    )
    text_features = normalize_features(
        encode_texts(model, tokenizer, captions, batch_size)
    )

    image_to_text_sim = image_features @ text_features.T
    text_to_image_sim = text_features @ image_features.T

    # Build positive sets: each image is positive for its 5 captions, and vice versa
    positives_i2t: List[List[int]] = [[] for _ in unique_images]
    for caption_idx, img_idx in enumerate(caption_image_indices):
        positives_i2t[img_idx].append(caption_idx)

    positives_t2i = [[caption_image_indices[idx]] for idx in range(len(captions))]

    ks = (1, 5, 10)
    metrics = {
        "image_to_text": compute_recall_at_k(image_to_text_sim.cpu().numpy(), positives_i2t, ks),
        "text_to_image": compute_recall_at_k(text_to_image_sim.cpu().numpy(), positives_t2i, ks),
    }
    return metrics


def main() -> None:
    args = parse_args()

    if not args.annotations.is_file():
        raise FileNotFoundError(f"annotations file not found: {args.annotations}")
    if not args.image_root.is_dir():
        raise NotADirectoryError(f"image root is not a directory: {args.image_root}")
    if not args.finetuned_checkpoint.is_file():
        raise FileNotFoundError(f"fine-tuned checkpoint not found: {args.finetuned_checkpoint}")
    if args.baseline_checkpoint and not args.baseline_checkpoint.is_file():
        raise FileNotFoundError(f"baseline checkpoint not found: {args.baseline_checkpoint}")

    df = pd.read_csv(args.annotations)
    if "image" not in df.columns or "caption" not in df.columns:
        raise ValueError("annotations CSV must contain 'image' and 'caption' columns")
    if args.split and "split" not in df.columns:
        print("[warn] Split specified but no 'split' column found; using the entire dataset")

    eval_split: str | None
    if args.split and args.split.lower() != "all":
        eval_split = args.split
    else:
        eval_split = None

    model_base, preprocess_base, tokenizer_base = load_model(
        args.model_name, args.pretrained, args.baseline_checkpoint
    )
    model_ft, preprocess_ft, tokenizer_ft = load_model(
        args.model_name, args.pretrained, args.finetuned_checkpoint
    )

    print("[info] Evaluating baseline model...")
    baseline_metrics = evaluate(
        model_base,
        preprocess_base,
        tokenizer_base,
        df,
        args.image_root,
        args.batch_size,
        eval_split,
    )

    print("[info] Evaluating fine-tuned model...")
    finetuned_metrics = evaluate(
        model_ft,
        preprocess_ft,
        tokenizer_ft,
        df,
        args.image_root,
        args.batch_size,
        eval_split,
    )

    comparison = {
        "baseline": baseline_metrics,
        "fine_tuned": finetuned_metrics,
    }

    print("\nBaseline vs Fine-Tuned Recall@K (%):")
    for label, metrics in comparison.items():
        print(f"\n=== {label} ===")
        for direction, stats in metrics.items():
            formatted = ", ".join(f"R@{k}: {stats[f'R@{k}']*100:.2f}" for k in (1, 5, 10))
            print(f"{direction}: {formatted}")

    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w", encoding="utf-8") as f:
            json.dump(comparison, f, indent=2)
        print(f"[info] Wrote metrics to {args.output_json}")


if __name__ == "__main__":
    main()
