#!/usr/bin/env python3
"""Create proper train/val/test splits for Flickr30k dataset."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

def create_flickr30k_splits():
    """Create standard Flickr30k train/val/test splits."""
    
    # Load the full annotations
    annotations_file = "data/flickr30k_annotations.csv"
    df = pd.read_csv(annotations_file)
    
    print(f"Total captions: {len(df)}")
    print(f"Unique images: {df['image'].nunique()}")
    
    # Get unique images and create splits
    unique_images = df['image'].unique()
    np.random.seed(42)  # For reproducibility
    np.random.shuffle(unique_images)
    
    total_images = len(unique_images)
    print(f"Total unique images: {total_images}")
    
    # Standard Flickr30k splits: 29k train, 1k val, 1k test
    # But we'll use proportional splits for our dataset size
    train_size = int(0.8 * total_images)  # 80% train
    val_size = int(0.1 * total_images)    # 10% val
    test_size = total_images - train_size - val_size  # 10% test
    
    train_images = set(unique_images[:train_size])
    val_images = set(unique_images[train_size:train_size + val_size])
    test_images = set(unique_images[train_size + val_size:])
    
    print(f"Split sizes:")
    print(f"  Training: {len(train_images)} images")
    print(f"  Validation: {len(val_images)} images")
    print(f"  Test: {len(test_images)} images")
    
    # Assign splits to dataframe
    def assign_split(image_name):
        if image_name in train_images:
            return 'train'
        elif image_name in val_images:
            return 'val'
        else:
            return 'test'
    
    df['split'] = df['image'].apply(assign_split)
    
    # Save split datasets
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'val'].copy()
    test_df = df[df['split'] == 'test'].copy()
    
    # Create split directory
    split_dir = Path("data/splits")
    split_dir.mkdir(exist_ok=True)
    
    # Save individual split files
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    
    # Save the full annotated dataset
    df.to_csv("data/flickr30k_annotations_with_splits.csv", index=False)
    
    print(f"\\nSplit statistics:")
    for split in ['train', 'val', 'test']:
        split_df = df[df['split'] == split]
        print(f"  {split}: {len(split_df)} captions, {split_df['image'].nunique()} images")
    
    # Save split info
    split_info = {
        'total_images': total_images,
        'total_captions': len(df),
        'splits': {
            'train': {'images': len(train_images), 'captions': len(train_df)},
            'val': {'images': len(val_images), 'captions': len(val_df)},
            'test': {'images': len(test_images), 'captions': len(test_df)}
        },
        'train_images': list(train_images)[:10],  # Sample for verification
        'val_images': list(val_images)[:10],
        'test_images': list(test_images)[:10]
    }
    
    with open("data/splits/split_info.json", 'w') as f:
        json.dump(split_info, f, indent=2)
    
    print(f"\\nSplit files saved to {split_dir}/")
    print("Files created:")
    print("  - train.csv")
    print("  - val.csv") 
    print("  - test.csv")
    print("  - split_info.json")
    print("  - ../flickr30k_annotations_with_splits.csv")
    
    return split_info

if __name__ == "__main__":
    split_info = create_flickr30k_splits()