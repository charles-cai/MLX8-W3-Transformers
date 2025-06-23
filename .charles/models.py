import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np

from datasets import load_dataset

# class MNISTDataset(Dataset):
#     def __init__(self, parquet_files):
#         # Load all parquet files and combine them
#         dfs = []
#         for file in parquet_files:
#             df = pd.read_parquet(file)
#             dfs.append(df)
        
#         if dfs:
#             self.data = pd.concat(dfs, ignore_index=True)
#         else:
#             raise ValueError(f"No parquet files found: {parquet_files}")
        
#         print(f"Loaded {len(self.data)} samples")
        
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, idx):
#         row = self.data.iloc[idx]
        
#         # Handle image data - MNIST in HF parquet format stores images differently
#         image_data = row['image']
        
#         # Check if image is stored as a dictionary with 'bytes' key
#         if isinstance(image_data, dict) and 'bytes' in image_data:
#             # Extract bytes and convert to numpy array
#             image_bytes = image_data['bytes']
#             if isinstance(image_bytes, bytes):
#                 image = np.frombuffer(image_bytes, dtype=np.uint8)
#             else:
#                 image = np.array(image_bytes, dtype=np.uint8)
#         elif isinstance(image_data, list):
#             image = np.array(image_data, dtype=np.uint8)
#         elif hasattr(image_data, 'numpy'):  # PIL Image or similar
#             image = np.array(image_data)
#         else:
#             image = np.array(image_data, dtype=np.uint8)
        
#         # Debug print to understand the data structure
#         if idx == 0:  # Print only for first sample
#             print(f"Image data type: {type(image_data)}")
#             if isinstance(image_data, dict):
#                 print(f"Image dict keys: {image_data.keys()}")
#             print(f"Processed image shape: {image.shape}")
#             print(f"Image min/max: {image.min()}/{image.max()}")
        
#         # Reshape to 28x28 if it's flattened (MNIST is 784 pixels = 28*28)
#         if len(image.shape) == 1 and len(image) == 784:
#             image = image.reshape(28, 28)
#         elif len(image.shape) == 1:
#             print(f"Warning: Unexpected flattened image size: {len(image)}")
#             # Try to infer square dimensions
#             size = int(np.sqrt(len(image)))
#             if size * size == len(image):
#                 image = image.reshape(size, size)
#             else:
#                 raise ValueError(f"Cannot reshape image of size {len(image)} to square")
        
#         # Convert to tensor and normalize
#         image = torch.tensor(image, dtype=torch.float32) / 255.0
#         label = torch.tensor(row['label'], dtype=torch.long)
        
#         return {'image': image, 'label': label}

def get_mnist_data_loaders(batch_size_train=128, batch_size_test=256):

    train_files = ".data/ylecun/mnist/train*.parquet"
    test_files = ".data/ylecun/mnist/test*.parquet"

    # 1. Load from local directory (no download)
    train_ds = load_dataset("parquet", data_files=train_files)["train"]
    test_ds = load_dataset("parquet", data_files=test_files)["train"]

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

    train_loader = DataLoader(
        train_ds, batch_size=batch_size_train, shuffle=True,
        num_workers=4, pin_memory=True)

    test_loader = DataLoader(
        test_ds, batch_size=batch_size_test, shuffle=False,
        num_workers=2)
    
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
    def __init__(self, d_model=128, n_heads=4, depth=4, patch=4):
        super().__init__()
        self.patch = PatchEmbed(patch, d_model)
        # Fix: Account for CLS token in positioning (49 patches + 1 CLS = 50 tokens)
        self.pos   = LearnablePos(50, d_model)
        
        # Option 1: Use custom transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout=0.1) 
            for _ in range(depth)
        ])
        
        # Option 2: Keep PyTorch's built-in (commented out)
        # enc_layer  = nn.TransformerEncoderLayer(
        #                 d_model, n_heads, 4*d_model, 0.1, batch_first=True)
        # self.encoder = nn.TransformerEncoder(enc_layer, depth)

        # Use a dedicated [CLS] token to aggregate sequence information
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

def train(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VitMNISTEncoder().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  # Add epoch loop
        total_loss = 0
        # Wrap train_loader with tqdm for progress bar
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}")):
            # Fix: Handle batch structure - image is already normalized and needs channel dimension
            img = batch['image'].unsqueeze(1)  # Add channel dimension: (B, 28, 28) -> (B, 1, 28, 28)
            label = batch['label']
            
            img, label = img.to(device), label.to(device)
            
            optim.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')
    
        print(f'Epoch {epoch} completed, Average Loss: {total_loss/len(train_loader):.4f}')
    
    # Evaluation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            img = batch['image'].unsqueeze(1)  # Add channel dimension
            label = batch['label']
            img, label = img.to(device), label.to(device)
            
            logits = model(img)
            _, predicted = torch.max(logits.data, 1)
            total += label.size(0)
            correct += (predicted == label).sum().item()
    
    print(f'Test Accuracy: {100 * correct / total:.2f}%')

# Usage example
if __name__ == "__main__":
    train_loader, test_loader = get_mnist_data_loaders()
    train(train_loader=train_loader, test_loader=test_loader)