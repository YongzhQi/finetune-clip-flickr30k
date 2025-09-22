#!/usr/bin/env python3
"""Proper CLIP fine-tuning with validation and early stopping."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import open_clip
import json
import math
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import pandas as pd

class FlickrDataset(Dataset):
    def __init__(self, csv_file, image_root, preprocess, tokenizer, max_samples=None):
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.head(max_samples)
        self.image_root = Path(image_root)
        self.preprocess = preprocess
        self.tokenizer = tokenizer
        
        # Cache tokenized captions for efficiency
        print(f"Tokenizing {len(self.df)} captions...")
        self.tokenized_captions = []
        for caption in tqdm(self.df['caption'], desc="Tokenizing"):
            tokens = self.tokenizer([caption])[0]
            self.tokenized_captions.append(tokens)
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = self.image_root / row['image']
        
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert('RGB')
            image = self.preprocess(image)
            
            # Get pre-tokenized caption
            tokens = self.tokenized_captions[idx]
            
            return {
                'image': image,
                'tokens': tokens,
                'caption': row['caption'],
                'image_name': row['image']
            }
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy sample
            dummy_image = torch.zeros(3, 224, 224)
            dummy_tokens = torch.zeros(77, dtype=torch.long)
            return {
                'image': dummy_image,
                'tokens': dummy_tokens,
                'caption': "dummy",
                'image_name': "dummy.jpg"
            }

def collate_fn(batch):
    """Custom collate function to handle batching."""
    images = torch.stack([item['image'] for item in batch])
    tokens = torch.stack([item['tokens'] for item in batch])
    captions = [item['caption'] for item in batch]
    image_names = [item['image_name'] for item in batch]
    
    return {
        'image': images,
        'tokens': tokens,
        'captions': captions,
        'image_names': image_names
    }

def compute_contrastive_loss(model, images, tokens, device='cpu'):
    """Compute symmetric contrastive loss."""
    # Get features
    image_features = model.encode_image(images)
    text_features = model.encode_text(tokens)
    
    # Normalize features
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    # Get logit scale
    logit_scale = model.logit_scale.exp()
    
    # Compute similarities
    logits_per_image = logit_scale * image_features @ text_features.t()
    logits_per_text = logit_scale * text_features @ image_features.t()
    
    # Labels for contrastive loss (diagonal should be positive)
    batch_size = images.shape[0]
    labels = torch.arange(batch_size, device=device)
    
    # Symmetric loss
    loss_i2t = nn.functional.cross_entropy(logits_per_image, labels)
    loss_t2i = nn.functional.cross_entropy(logits_per_text, labels)
    loss = (loss_i2t + loss_t2i) / 2
    
    return loss, logits_per_image, logits_per_text

def train_one_epoch(model, dataloader, optimizer, device='cpu', max_batches=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    batches = 0
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        optimizer.zero_grad()
        
        images = batch['image'].to(device)
        tokens = batch['tokens'].to(device)
        
        try:
            loss, _, _ = compute_contrastive_loss(model, images, tokens, device)
            
            loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Clamp logit scale
            with torch.no_grad():
                model.logit_scale.data.clamp_(math.log(1 / 0.07), math.log(100.0))
            
            total_loss += loss.item()
            batches += 1
            
            # Update progress bar
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'avg_loss': f'{total_loss/batches:.4f}'})
            
            if max_batches and batches >= max_batches:
                print(f"\\nStopping at {batches} batches (max_batches limit)")
                break
                
        except Exception as e:
            print(f"Error in batch {batches}: {e}")
            continue
    
    return total_loss / max(batches, 1)

def validate_model(model, dataloader, device='cpu', max_batches=None):
    """Validate the model."""
    model.eval()
    total_loss = 0
    batches = 0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validating")
        for batch in pbar:
            images = batch['image'].to(device)
            tokens = batch['tokens'].to(device)
            
            try:
                loss, _, _ = compute_contrastive_loss(model, images, tokens, device)
                total_loss += loss.item()
                batches += 1
                
                pbar.set_postfix({'val_loss': f'{loss.item():.4f}'})
                
                if max_batches and batches >= max_batches:
                    break
                    
            except Exception as e:
                print(f"Error in validation batch {batches}: {e}")
                continue
    
    return total_loss / max(batches, 1)

def main():
    print("Setting up improved CLIP fine-tuning...")
    
    # Configuration
    config = {
        'model_name': 'ViT-B-32',
        'pretrained': 'openai',
        'batch_size': 32,  # Larger batch size
        'learning_rate': 5e-5,  # Much lower learning rate
        'weight_decay': 0.1,
        'epochs': 3,
        'device': 'cpu',
        'max_train_batches': 200,  # Limit for CPU training
        'max_val_batches': 50,
        'train_max_samples': 6400,  # 200 batches * 32 batch_size
        'val_max_samples': 1600    # 50 batches * 32 batch_size
    }
    
    print(f"Config: {json.dumps(config, indent=2)}")
    
    # Load model
    print("Loading CLIP model...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        config['model_name'], pretrained=config['pretrained'], device=config['device']
    )
    tokenizer = open_clip.get_tokenizer(config['model_name'])
    
    # Freeze all parameters except projection heads
    for param in model.parameters():
        param.requires_grad = False
    
    # Enable gradients for projection layers only
    trainable_params = []
    if hasattr(model, 'text_projection') and model.text_projection is not None:
        model.text_projection.requires_grad = True
        trainable_params.append(model.text_projection)
        print("Enabled training for text_projection")
    
    if hasattr(model, 'visual') and hasattr(model.visual, 'proj') and model.visual.proj is not None:
        model.visual.proj.requires_grad = True
        trainable_params.append(model.visual.proj)
        print("Enabled training for visual.proj")
    
    model.logit_scale.requires_grad = True
    trainable_params.append(model.logit_scale)
    print("Enabled training for logit_scale")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_param_count:,}")
    print(f"Training {trainable_param_count/total_params*100:.2f}% of parameters")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset = FlickrDataset(
        csv_file="data/splits/train.csv",
        image_root="data/flickr30k_images",
        preprocess=preprocess,
        tokenizer=tokenizer,
        max_samples=config['train_max_samples']
    )
    
    val_dataset = FlickrDataset(
        csv_file="data/splits/val.csv",
        image_root="data/flickr30k_images",
        preprocess=preprocess,
        tokenizer=tokenizer,
        max_samples=config['val_max_samples']
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    print(f"Training: {len(train_dataset)} samples, {len(train_loader)} batches")
    print(f"Validation: {len(val_dataset)} samples, {len(val_loader)} batches")
    
    # Set up optimizer with lower learning rate
    optimizer = optim.AdamW(
        trainable_params,
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config['epochs']
    )
    
    # Create output directories
    Path("artifacts").mkdir(exist_ok=True)
    Path("checkpoints").mkdir(exist_ok=True)
    
    # Training loop
    print("Starting training...")
    best_val_loss = float('inf')
    training_history = []
    
    for epoch in range(config['epochs']):
        print(f"\\nEpoch {epoch + 1}/{config['epochs']}")
        
        # Training
        train_loss = train_one_epoch(
            model, train_loader, optimizer, config['device'], 
            max_batches=config['max_train_batches']
        )
        
        # Validation
        val_loss = validate_model(
            model, val_loader, config['device'],
            max_batches=config['max_val_batches']
        )
        
        # Learning rate step
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        
        print(f"Epoch {epoch + 1} - Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, LR: {current_lr:.2e}")
        
        # Save checkpoint if validation improved
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = "checkpoints/linearprobe_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
                'config': config,
                'best_val_loss': best_val_loss
            }, checkpoint_path)
            print(f"✓ Saved best model (val_loss: {val_loss:.4f})")
        
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss,
            'lr': current_lr,
            'best_val_loss': best_val_loss
        })
    
    # Save training history
    with open("artifacts/training_history.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Best model saved to: checkpoints/linearprobe_best.pt")
    print(f"Training history saved to: artifacts/training_history.json")
    
    return checkpoint_path, training_history

if __name__ == "__main__":
    checkpoint_path, history = main()