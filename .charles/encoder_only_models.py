import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import wandb

from datasets import load_dataset
from wandb_utils import WandbLogger

# Load environment variables
load_dotenv()

def get_mnist_data_loaders(batch_size_train=128, batch_size_test=256):
    print("="*60)
    print("STARTING DATA PREPARATION - ENCODER ONLY")
    print("="*60)
    
    # Get batch sizes from environment variables
    batch_size_train = int(os.getenv('BATCH_SIZE_TRAIN', '128'))
    batch_size_test = int(os.getenv('BATCH_SIZE_TEST', '256'))
    
    train_files = f"{os.getenv('MNIST_DATA_PATH', '.data/ylecun/mnist')}/train*.parquet"
    test_files = f"{os.getenv('MNIST_DATA_PATH', '.data/ylecun/mnist')}/test*.parquet"

    print(f"Loading MNIST data from:")
    print(f"  Train files: {train_files}")
    print(f"  Test files: {test_files}")

    # 1. Load from local directory (no download)
    print("Loading single digit datasets...")
    train_ds = load_dataset("parquet", data_files=train_files)["train"]
    test_ds = load_dataset("parquet", data_files=test_files)["train"]
    print(f"Loaded {len(train_ds)} training samples and {len(test_ds)} test samples")

    # Transform function to normalize images
    def transform_batch(batch):
        # Convert PIL images to numpy arrays, then to tensors and normalize to [0, 1]
        images = []
        for img in batch['image']:
            if hasattr(img, 'numpy'):  # PIL Image
                img_array = np.array(img)
            elif isinstance(img, np.ndarray):
                img_array = img
            else:
                # Try to convert to numpy array
                img_array = np.array(img)
            
            # Convert to float32 tensor and normalize
            img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
            images.append(img_tensor)
        
        images = torch.stack(images)
        labels = torch.tensor(batch['label'], dtype=torch.long)
        return {'image': images, 'label': labels}

    train_ds = train_ds.with_transform(transform_batch)
    test_ds = test_ds.with_transform(transform_batch)

    print(f"Creating data loaders:")
    print(f"  Training batch size: {batch_size_train}")
    print(f"  Test batch size: {batch_size_test}")

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size_test, shuffle=False,
        num_workers=2)
    
    print("Data preparation completed!")
    print("="*60)
    
    return train_loader, test_loader

class PatchEmbed(nn.Module):
    """Turn 28×28 image → sequence of 49 tokens (patch=4)."""
    def __init__(self, patch=4, d_model=128):
        super().__init__()
        self.proj = nn.Conv2d(1, d_model, patch, patch)   # (B,1,28,28) → (B,d,7,7)

    def forward(self, x):
        x = self.proj(x)                                  # (B,d,7,7)
        return x.flatten(2).transpose(1, 2)               # (B,49,d_model)

class LearnablePos(nn.Module):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))

    def forward(self, x):
        return x + self.pos

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, N, D = x.shape
        
        # Generate Q, K, V
        Q = self.w_q(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)  # (B, h, N, d_k)
        K = self.w_k(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)  # (B, h, N, d_k)
        
        # Concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, N, D)
        return self.w_o(attn_output)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        return self.linear2(self.dropout(torch.relu(self.linear1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention with residual connection
        attn_output = self.attention(x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class VitMNISTEncoder(nn.Module):
    """
    [PatchEmbed] → [Pos] → [Encoder×L] → optional [CLS] pooling → logits(10)
    """
    def __init__(self, d_model=128, n_heads=4, depth=4, patch=4, dropout=0.1):
        super().__init__()
        # Store config for checkpointing
        self.d_model = d_model
        self.n_heads = n_heads 
        self.depth = depth
        self.patch_size = patch
        
        self.patch = PatchEmbed(patch, d_model)
        # Fix: Account for CLS token in positioning (49 patches + 1 CLS = 50 tokens)
        self.pos   = LearnablePos(50, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=dropout) 
            for _ in range(depth)
        ])

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, 10)

    def forward(self, img):                      # img: (B,1,28,28)
        B = img.size(0)
        x = self.patch(img)                      # (B,49,d_model)
        
        cls = self.cls_token.expand(B, -1, -1)   # (B,1,d_model)
        x   = torch.cat([cls, x], dim=1)         # (B,50,d_model)
        
        # Apply positional embedding to the full sequence (CLS + patches)
        x = self.pos(x)

        # Apply transformer blocks
        for block in self.transformer_blocks:
            x = block(x)                         # (B,50,d_model)
        
        # Use PyTorch's built-in encoder (alternative)
        # z = self.encoder(x)                    # (B,50,d_model)
        
        logits = self.head(x[:, 0])              # take [CLS] (index 0)
        return logits

    def save(self, filepath, epoch, optimizer=None, loss=None, accuracy=None):
        """Save model checkpoint"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'loss': loss,
            'accuracy': accuracy,
            'model_config': {
                'd_model': getattr(self, 'd_model', 128),
                'n_heads': getattr(self, 'n_heads', 4),
                'depth': getattr(self, 'depth', 4),
                'patch': getattr(self, 'patch', 4)
            }
        }
        
        if optimizer is not None:
            checkpoint['optimizer_state_dict'] = optimizer.state_dict()
            
        torch.save(checkpoint, filepath)
        print(f"Model checkpoint saved to {filepath}")

def train(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Check if running in wandb sweep mode
    if wandb.run is not None:
        # Use wandb config (sweep mode)
        config = wandb.config
        learning_rate = config.learning_rate
        num_epochs = config.get('num_epochs', 10)
        d_model = config.get('d_model', 128)
        n_heads = config.get('n_heads', 4)
        depth = config.get('depth', 4)
        patch_size = config.get('patch_size', 4)
        dropout = config.get('dropout', 0.1)
        print("Running in wandb sweep mode")
    else:
        # Use environment variables (default mode)
        learning_rate = float(os.getenv('LEARNING_RATE', '3e-4'))
        num_epochs = int(os.getenv('NUM_EPOCHS', '8'))
        d_model = 128
        n_heads = 4
        depth = 4
        patch_size = 4
        dropout = 0.1
        print("Running in default mode from .env")
    
    models_folder = os.getenv('MODELS_FOLDER', '.data/models')
    
    # Initialize model with parameters
    model = VitMNISTEncoder(
        d_model=d_model,
        n_heads=n_heads,
        depth=depth,
        patch=patch_size,
        dropout=dropout
    ).to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Update config for logging
    wandb_config = {
        'learning_rate': learning_rate,
        'epochs': num_epochs,
        'batch_size_train': train_loader.batch_size,
        'batch_size_test': test_loader.batch_size,
        'd_model': d_model,
        'n_heads': n_heads,
        'depth': depth,
        'patch_size': patch_size,
        'dropout': dropout,
        'device': str(device),
        'model_params': sum(p.numel() for p in model.parameters()),
    }
    
    # Initialize wandb logger only if not already in sweep
    if wandb.run is None:
        run_name = os.getenv('WANDB_RUN_NAME_ENCODER_ONLY', 'vit-mnist-encoder-only')
        wandb_logger = WandbLogger(wandb_config, run_name=run_name)
        wandb_logger.log_model(model)
    else:
        # In sweep mode, wandb is already initialized
        wandb_logger = WandbLogger(config=None)  # Don't reinitialize
        wandb.watch(model, log="all")

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        # Training loop
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", 
                                               unit="batch", 
                                               postfix={'loss': 0, 'acc': 0})):
            img = batch['image'].unsqueeze(1)
            label = batch['label']
            
            img, label = img.to(device), label.to(device)
            
            optim.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            
            # Calculate training accuracy
            _, predicted = torch.max(logits.data, 1)
            total_train += label.size(0)
            correct_train += (predicted == label).sum().item()
            
            # Update tqdm postfix with current metrics
            if batch_idx % 50 == 0:
                current_acc = 100 * correct_train / total_train if total_train > 0 else 0
                tqdm.write(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}, Acc: {current_acc:.2f}%')
                
                # Log batch metrics to wandb
                wandb_logger.log_metrics({
                    'batch_loss': loss.item(),
                    'batch_idx': batch_idx,
                    'epoch': epoch + 1
                })
        
        # Calculate epoch metrics
        avg_train_loss = total_loss / len(train_loader)
        train_accuracy = 100 * correct_train / total_train
        
        # Evaluate on test set
        model.eval()
        test_loss = 0
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for batch in test_loader:
                img = batch['image'].unsqueeze(1)
                label = batch['label']
                img, label = img.to(device), label.to(device)
                
                logits = model(img)
                loss = criterion(logits, label)
                test_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total_test += label.size(0)
                correct_test += (predicted == label).sum().item()
        
        avg_test_loss = test_loss / len(test_loader)
        test_accuracy = 100 * correct_test / total_test
        
        print(f'Epoch {epoch+1} completed:')
        print(f'  Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%')
        print(f'  Test Loss: {avg_test_loss:.4f}, Test Accuracy: {test_accuracy:.2f}%')
        
        # Log epoch metrics to wandb
        wandb_logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_train_loss,
            'train_accuracy': train_accuracy,
            'test_loss': avg_test_loss,
            'test_accuracy': test_accuracy,
        })
        
        # Save model checkpoint
        checkpoint_path = os.path.join(models_folder, f'vit_mnist_epoch_{epoch + 1}.pth')
        model.save(checkpoint_path, epoch + 1, optim, avg_test_loss, test_accuracy)
        
        # Log model artifact to wandb
        wandb_logger.save_artifact(checkpoint_path, f'model_epoch_{epoch + 1}')
        
        model.train()  # Set back to training mode
    
    print(f'Training completed. Final Test Accuracy: {test_accuracy:.2f}%')
    wandb_logger.finish()

# Usage example
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_data_loaders()
    train(train_loader=train_loader, test_loader=test_loader)