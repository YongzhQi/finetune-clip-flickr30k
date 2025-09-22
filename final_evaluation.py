#!/usr/bin/env python3
"""Final evaluation of improved fine-tuned CLIP on held-out test set."""

import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import open_clip

def load_model(checkpoint_path=None):
    """Load either baseline or fine-tuned model."""
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="openai", device="cpu"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    
    if checkpoint_path:
        # Load checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
        print(f"Training loss: {checkpoint.get('train_loss', 'unknown'):.4f}")
        print(f"Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
    
    model.eval()
    return model, preprocess, tokenizer

def encode_images(model, preprocess, image_paths, batch_size=16):
    """Encode images using CLIP."""
    features = []
    failed_images = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding images"):
            batch_paths = image_paths[i:i+batch_size]
            images = []
            
            for path in batch_paths:
                try:
                    with Image.open(path) as img:
                        image = img.convert("RGB")
                    images.append(preprocess(image))
                except Exception as e:
                    failed_images += 1
                    if failed_images <= 3:  # Only print first few errors
                        print(f"Error loading {path}: {e}")
                    continue
            
            if images:
                image_batch = torch.stack(images)
                feat = model.encode_image(image_batch)
                features.append(feat.cpu())
    
    if failed_images > 0:
        print(f"Failed to load {failed_images} images")
    
    return torch.cat(features, dim=0) if features else torch.empty(0)

def encode_texts(model, tokenizer, captions, batch_size=32):
    """Encode text captions using CLIP."""
    features = []
    with torch.no_grad():
        for i in tqdm(range(0, len(captions), batch_size), desc="Encoding captions"):
            batch_captions = captions[i:i+batch_size]
            tokens = tokenizer(batch_captions)
            feat = model.encode_text(tokens)
            features.append(feat.cpu())
    
    return torch.cat(features, dim=0)

def normalize_features(feats):
    """Normalize features to unit length."""
    return feats / feats.norm(dim=-1, keepdim=True).clamp_min(1e-12)

def compute_recall_at_k(similarity, positives, ks=[1, 5, 10]):
    """Compute Recall@K metrics."""
    if not positives:
        return {f"R@{k}": float("nan") for k in ks}

    recalls = {}
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

def evaluate_on_test_set(model, preprocess, tokenizer, test_csv, image_root, max_samples=None):
    """Evaluate model on test set."""
    print(f"Loading test data from {test_csv}")
    df = pd.read_csv(test_csv)
    
    if max_samples:
        df = df.head(max_samples)
        print(f"Using subset of {max_samples} samples for faster evaluation")
    
    print(f"Test set: {len(df)} captions from {df['image'].nunique()} unique images")
    
    # Get unique images and build mappings
    unique_images = df["image"].unique().tolist()
    image_idx = {img: idx for idx, img in enumerate(unique_images)}
    
    # Prepare data
    image_paths = [image_root / img for img in unique_images]
    captions = df["caption"].tolist()
    caption_image_indices = df["image"].map(image_idx).tolist()
    
    print("Encoding features...")
    
    # Encode features
    image_features = normalize_features(
        encode_images(model, preprocess, image_paths, batch_size=8)
    )
    text_features = normalize_features(
        encode_texts(model, tokenizer, captions, batch_size=32)
    )
    
    print(f"Image features shape: {image_features.shape}")
    print(f"Text features shape: {text_features.shape}")
    
    # Compute similarities
    print("Computing similarities...")
    image_to_text_sim = (image_features @ text_features.T).numpy()
    text_to_image_sim = (text_features @ image_features.T).numpy()
    
    # Build positive sets for evaluation
    print("Building positive sets...")
    positives_i2t = [[] for _ in unique_images]
    for caption_idx, img_idx in enumerate(caption_image_indices):
        positives_i2t[img_idx].append(caption_idx)
    
    positives_t2i = [[caption_image_indices[idx]] for idx in range(len(captions))]
    
    # Compute metrics
    print("Computing recall metrics...")
    metrics = {
        "image_to_text": compute_recall_at_k(image_to_text_sim, positives_i2t),
        "text_to_image": compute_recall_at_k(text_to_image_sim, positives_t2i),
    }
    
    return metrics

def main():
    # Configuration
    test_csv = "data/splits/test.csv"
    image_root = Path("data/flickr30k_images")
    baseline_checkpoint = None  # Use pretrained
    finetuned_checkpoint = "checkpoints/linearprobe_best.pt"
    
    # Use subset for faster evaluation (remove for full evaluation)
    max_test_samples = 2000  # ~400 images with 5 captions each
    
    print("="*60)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*60)
    
    # Evaluate baseline model
    print("\\n1. Evaluating BASELINE (pretrained CLIP)...")
    baseline_model, baseline_preprocess, baseline_tokenizer = load_model()
    baseline_metrics = evaluate_on_test_set(
        baseline_model, baseline_preprocess, baseline_tokenizer,
        test_csv, image_root, max_test_samples
    )
    
    # Evaluate fine-tuned model
    print("\\n2. Evaluating FINE-TUNED model...")
    finetuned_model, finetuned_preprocess, finetuned_tokenizer = load_model(finetuned_checkpoint)
    finetuned_metrics = evaluate_on_test_set(
        finetuned_model, finetuned_preprocess, finetuned_tokenizer,
        test_csv, image_root, max_test_samples
    )
    
    # Compare results
    print("\\n" + "="*60)
    print("RESULTS COMPARISON")
    print("="*60)
    
    print("\\nBaseline (Pretrained CLIP):")
    for direction, stats in baseline_metrics.items():
        formatted = ", ".join(f"R@{k}: {stats[f'R@{k}']*100:.2f}%" for k in [1, 5, 10])
        print(f"  {direction}: {formatted}")
    
    print("\\nFine-tuned (Linear Probe):")
    for direction, stats in finetuned_metrics.items():
        formatted = ", ".join(f"R@{k}: {stats[f'R@{k}']*100:.2f}%" for k in [1, 5, 10])
        print(f"  {direction}: {formatted}")
    
    # Compute improvements
    print("\\n" + "="*60)
    print("IMPROVEMENT ANALYSIS")
    print("="*60)
    
    improvements = {}
    for direction in ['image_to_text', 'text_to_image']:
        improvements[direction] = {}
        print(f"\\n{direction.replace('_', '→').upper()}:")
        for k in [1, 5, 10]:
            baseline_val = baseline_metrics[direction][f'R@{k}'] * 100
            finetuned_val = finetuned_metrics[direction][f'R@{k}'] * 100
            improvement_points = finetuned_val - baseline_val
            improvement_percent = (improvement_points / baseline_val) * 100 if baseline_val > 0 else 0
            
            improvements[direction][f'R@{k}'] = {
                'baseline': baseline_val,
                'finetuned': finetuned_val,
                'improvement_points': improvement_points,
                'improvement_percent': improvement_percent
            }
            
            print(f"  R@{k}: {baseline_val:.2f}% → {finetuned_val:.2f}% ({improvement_points:+.2f}pp, {improvement_percent:+.1f}%)")
    
    # Answer the original question
    print("\\n" + "="*60)
    print("ANSWER TO YOUR QUESTION:")
    print("="*60)
    
    t2i_r5 = improvements['text_to_image']['R@5']
    i2t_r5 = improvements['image_to_text']['R@5']
    
    print(f"\\n📊 Text→Image Recall@5:")
    print(f"   Baseline:     {t2i_r5['baseline']:.2f}%")
    print(f"   Fine-tuned:   {t2i_r5['finetuned']:.2f}%")
    print(f"   Improvement:  {t2i_r5['improvement_points']:+.2f} percentage points")
    print(f"   Relative:     {t2i_r5['improvement_percent']:+.1f}% change")
    
    print(f"\\n📊 Image→Text Recall@5:")
    print(f"   Baseline:     {i2t_r5['baseline']:.2f}%")
    print(f"   Fine-tuned:   {i2t_r5['finetuned']:.2f}%")
    print(f"   Improvement:  {i2t_r5['improvement_points']:+.2f} percentage points")
    print(f"   Relative:     {i2t_r5['improvement_percent']:+.1f}% change")
    
    # Summary verdict
    if t2i_r5['improvement_points'] > 0 or i2t_r5['improvement_points'] > 0:
        print(f"\\n✅ CONCLUSION: Fine-tuning achieved improvements!")
        if t2i_r5['improvement_points'] > 0:
            print(f"   • Text→Image improved by {t2i_r5['improvement_points']:.2f} points")
        if i2t_r5['improvement_points'] > 0:
            print(f"   • Image→Text improved by {i2t_r5['improvement_points']:.2f} points")
    else:
        print(f"\\n❌ CONCLUSION: Fine-tuning did not achieve improvements.")
        print(f"   • Both directions show decreased performance")
        print(f"   • May need longer training, different hyperparameters, or more data")
    
    # Save detailed results
    final_results = {
        'test_set_size': {
            'total_captions': max_test_samples if max_test_samples else 'full',
            'unique_images': len(baseline_metrics.get('image_to_text', {})) if baseline_metrics else 0
        },
        'baseline': baseline_metrics,
        'finetuned': finetuned_metrics,
        'improvements': improvements,
        'summary': {
            'text_to_image_r5_points': t2i_r5['improvement_points'],
            'image_to_text_r5_points': i2t_r5['improvement_points'],
            'text_to_image_r5_percent': t2i_r5['improvement_percent'],
            'image_to_text_r5_percent': i2t_r5['improvement_percent']
        }
    }
    
    results_file = "artifacts/final_evaluation_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"\\n💾 Detailed results saved to: {results_file}")
    
    return final_results

if __name__ == "__main__":
    results = main()