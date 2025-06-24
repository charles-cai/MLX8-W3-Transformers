import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import os
import glob
from tqdm import tqdm
import numpy as np
from dotenv import load_dotenv
import random

from datasets import load_dataset
from wandb_utils import WandbLogger

# Load environment variables
load_dotenv()

def create_4digit_dataset(single_digit_dataset, num_samples=10000):
    """Create dataset of 4 stacked digits from single MNIST digits"""
    samples = []
    
    for _ in range(num_samples):
        # Sample 4 random digits - THIS IS RANDOM, NOT SEQUENTIAL
        indices = random.choices(range(len(single_digit_dataset)), k=4)
        digits = []
        labels = []
        
        for idx in indices:
            sample = single_digit_dataset[idx]
            digits.append(sample['image'])
            labels.append(sample['label'])
        
        # Stack digits in 2x2 grid (56x56 total)
        top_row = torch.cat([digits[0], digits[1]], dim=1)  # (28, 56)
        bottom_row = torch.cat([digits[2], digits[3]], dim=1)  # (28, 56)
        stacked_image = torch.cat([top_row, bottom_row], dim=0)  # (56, 56)
        
        samples.append({
            'image': stacked_image,
            'labels': torch.tensor(labels, dtype=torch.long),  # [digit1, digit2, digit3, digit4]
        })
    
    return samples

def get_4digit_mnist_loaders(batch_size_train=64, batch_size_test=128):
    """Create data loaders for 4-digit stacked MNIST"""
    batch_size_train = int(os.getenv('BATCH_SIZE_TRAIN', '64'))
    batch_size_test = int(os.getenv('BATCH_SIZE_TEST', '128'))
    
    train_files = f"{os.getenv('MNIST_DATA_PATH', '.data/ylecun/mnist')}/train*.parquet"
    test_files = f"{os.getenv('MNIST_DATA_PATH', '.data/ylecun/mnist')}/test*.parquet"

    # Load single digit datasets
    train_ds = load_dataset("parquet", data_files=train_files)["train"]
    test_ds = load_dataset("parquet", data_files=test_files)["train"]

    def transform_batch(batch):
        images = []
        for img in batch['image']:
            img_array = np.array(img) if hasattr(img, 'numpy') else np.array(img)
            img_tensor = torch.tensor(img_array, dtype=torch.float32) / 255.0
            images.append(img_tensor)
        
        images = torch.stack(images)
        labels = torch.tensor(batch['label'], dtype=torch.long)
        return {'image': images, 'label': labels}

    train_ds = train_ds.with_transform(transform_batch)
    test_ds = test_ds.with_transform(transform_batch)
    
    # Get sample numbers from environment variables
    train_samples = int(os.getenv('TRAIN_4DIGIT_SAMPLES', '20000'))
    test_samples = int(os.getenv('TEST_4DIGIT_SAMPLES', '4000'))
    
    # Create 4-digit datasets with configurable sample sizes
    train_4digit = create_4digit_dataset(train_ds, num_samples=train_samples)
    test_4digit = create_4digit_dataset(test_ds, num_samples=test_samples)
    
    class FourDigitDataset(Dataset):
        def __init__(self, samples):
            self.samples = samples
            
        def __len__(self):
            return len(self.samples)
            
        def __getitem__(self, idx):
            return self.samples[idx]
    
    train_loader = DataLoader(FourDigitDataset(train_4digit), batch_size=batch_size_train, shuffle=True, num_workers=4)
    test_loader = DataLoader(FourDigitDataset(test_4digit), batch_size=batch_size_test, shuffle=False, num_workers=2)
    
    return train_loader, test_loader

class FourDigitPatchEmbed(nn.Module):
    """Turn 56×56 stacked image → sequence of patches for 4 digits"""
    def __init__(self, patch=7, d_model=256):
        super().__init__()
        # For 56x56 image with patch=7, we get 8x8=64 patches total
        self.proj = nn.Conv2d(1, d_model, patch, patch)
        self.patch = patch

    def forward(self, x):
        # x: (B, 1, 56, 56)
        x = self.proj(x)  # (B, d_model, 8, 8) 
        return x.flatten(2).transpose(1, 2)  # (B, 64, d_model)

class LearnablePos(nn.Module):
    def __init__(self, n_tokens, d_model):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, n_tokens, d_model))

    def forward(self, x):
        return x + self.pos

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, mask=None):
        B, N, D = query.shape
        
        Q = self.w_q(query).view(B, N, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(B, key.size(1), self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(B, value.size(1), self.n_heads, self.d_k).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
            
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        attn_output = torch.matmul(attn_weights, V)
        
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

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # Self-attention
        attn_output = self.attention(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff=None, dropout=0.1):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        # Masked self-attention
        attn_output = self.self_attention(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with encoder output
        cross_attn_output = self.cross_attention(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout(ff_output))
        
        return x

class FourDigitEncoder(nn.Module):
    """Encoder for 4-digit stacked images"""
    def __init__(self, d_model=256, n_heads=8, depth=6, patch=7):
        super().__init__()
        self.patch_embed = FourDigitPatchEmbed(patch, d_model)
        self.pos_embed = LearnablePos(64, d_model)  # 8x8 patches
        
        self.encoder_blocks = nn.ModuleList([
            TransformerEncoderBlock(d_model, n_heads, dropout=0.1)
            for _ in range(depth)
        ])
        
        # Project to 4 digit representations
        self.digit_proj = nn.Linear(d_model, d_model)
        
    def forward(self, img):
        # img: (B, 1, 56, 56)
        x = self.patch_embed(img)  # (B, 64, d_model)
        x = self.pos_embed(x)
        
        for block in self.encoder_blocks:
            x = block(x)
        
        # Average pool patches to get 4 digit representations
        # Reshape to 4 quadrants: (B, 4, 16, d_model) then pool
        B, _, D = x.shape
        x = x.view(B, 8, 8, D)
        
        # Extract 4 quadrants (each 4x4 patches)
        top_left = x[:, :4, :4, :].mean(dim=(1, 2))      # (B, d_model)
        top_right = x[:, :4, 4:, :].mean(dim=(1, 2))     # (B, d_model)
        bottom_left = x[:, 4:, :4, :].mean(dim=(1, 2))   # (B, d_model)
        bottom_right = x[:, 4:, 4:, :].mean(dim=(1, 2))  # (B, d_model)
        
        digit_encodings = torch.stack([top_left, top_right, bottom_left, bottom_right], dim=1)  # (B, 4, d_model)
        return self.digit_proj(digit_encodings)

class FourDigitDecoder(nn.Module):
    """Decoder for generating 4-digit sequence"""
    def __init__(self, d_model=256, n_heads=8, depth=6, vocab_size=11):  # 0-9 + special tokens
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Token embeddings (0-9 digits + start/end tokens)
        self.token_embed = nn.Embedding(vocab_size, d_model)
        self.pos_embed = LearnablePos(5, d_model)  # Changed back to 5 for generation
        
        self.decoder_blocks = nn.ModuleList([
            TransformerDecoderBlock(d_model, n_heads, dropout=0.1)
            for _ in range(depth)
        ])
        
        self.output_proj = nn.Linear(d_model, vocab_size)
        
    def forward(self, tgt_tokens, encoder_output, tgt_mask=None):
        # tgt_tokens: (B, seq_len)
        x = self.token_embed(tgt_tokens)  # (B, seq_len, d_model)
        
        # Handle variable sequence lengths by slicing positional embeddings
        seq_len = x.size(1)
        pos_embed = self.pos_embed.pos[:, :seq_len, :]  # Slice to match sequence length
        x = x + pos_embed
        
        for block in self.decoder_blocks:
            x = block(x, encoder_output, tgt_mask=tgt_mask)
        
        return self.output_proj(x)  # (B, seq_len, vocab_size)

def create_causal_mask(size):
    """Create causal mask for decoder self-attention"""
    mask = torch.triu(torch.ones(size, size), diagonal=1)
    return mask == 0

class EncoderDecoderModel(nn.Module):
    """Complete Encoder-Decoder model for 4-digit recognition"""
    def __init__(self, d_model=256, n_heads=8, enc_depth=6, dec_depth=6):
        super().__init__()
        self.encoder = FourDigitEncoder(d_model, n_heads, enc_depth)
        self.decoder = FourDigitDecoder(d_model, n_heads, dec_depth)
        self.d_model = d_model
        
    def forward(self, src_img, tgt_tokens):
        # Encode 4-digit image
        encoder_output = self.encoder(src_img)  # (B, 4, d_model)
        
        # Create causal mask for decoder
        seq_len = tgt_tokens.size(1)
        tgt_mask = create_causal_mask(seq_len).to(tgt_tokens.device)
        
        # Decode sequence
        decoder_output = self.decoder(tgt_tokens, encoder_output, tgt_mask)
        return decoder_output
    
    def generate(self, src_img, max_length=5, start_token=10):
        """Generate 4-digit sequence autoregressively"""
        self.eval()
        device = src_img.device
        B = src_img.size(0)
        
        # Encode image
        with torch.no_grad():
            encoder_output = self.encoder(src_img)
            
            # Start with start token
            generated = torch.full((B, 1), start_token, dtype=torch.long, device=device)
            
            for _ in range(4):  # Generate exactly 4 digits
                tgt_mask = create_causal_mask(generated.size(1)).to(device)
                logits = self.decoder(generated, encoder_output, tgt_mask)
                
                # Get next token (last position)
                next_token = torch.argmax(logits[:, -1, :10], dim=-1)  # Only consider digits 0-9
                generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        
        return generated[:, 1:]  # Remove start token, return only digits

def train_encoder_decoder(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Configuration
    models_folder = os.getenv('MODELS_FOLDER', '.data/models')
    learning_rate = float(os.getenv('LEARNING_RATE', '1e-4'))
    num_epochs = int(os.getenv('NUM_EPOCHS', '10'))
    
    # Initialize model
    model = EncoderDecoderModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    
    # Wandb config
    config = {
        'learning_rate': learning_rate,
        'epochs': num_epochs,
        'batch_size_train': train_loader.batch_size,
        'model_params': sum(p.numel() for p in model.parameters()),
        'architecture': 'encoder_decoder_4digit'
    }
    
    # Use encoder-decoder specific run name
    run_name = os.getenv('WANDB_RUN_NAME_ENCODER_DECODER', 'vit-mnist-encoder-decoder')
    wandb_logger = WandbLogger(config, run_name=run_name)
    
    wandb_logger.log_model(model)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_sequences = 0
        total_sequences = 0
        
        for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}")):
            img = batch['image'].unsqueeze(1).to(device)  # (B, 1, 56, 56)
            labels = batch['labels'].to(device)  # (B, 4)
            
            B = img.size(0)
            
            # Prepare decoder input (start token + labels without last)
            start_tokens = torch.full((B, 1), 10, dtype=torch.long, device=device)  # 10 is start token
            tgt_input = torch.cat([start_tokens, labels[:, :-1]], dim=1)  # (B, 4)
            tgt_output = labels  # (B, 4)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(img, tgt_input)  # (B, 4, 11)
            
            # Calculate loss
            loss = criterion(logits.reshape(-1, 11), tgt_output.reshape(-1))
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy (exact sequence match)
            pred_sequences = torch.argmax(logits[:, :, :10], dim=-1)  # Only digits 0-9
            correct_sequences += (pred_sequences == labels).all(dim=1).sum().item()
            total_sequences += B
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch+1}, Batch {batch_idx}, Loss: {loss.item():.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in test_loader:
                img = batch['image'].unsqueeze(1).to(device)
                labels = batch['labels'].to(device)
                
                # Generate sequences
                generated = model.generate(img)
                
                # Calculate accuracy
                val_correct += (generated == labels).all(dim=1).sum().item()
                val_total += img.size(0)
        
        train_acc = 100 * correct_sequences / total_sequences
        val_acc = 100 * val_correct / val_total
        avg_loss = total_loss / len(train_loader)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
        
        # Log metrics
        wandb_logger.log_metrics({
            'epoch': epoch + 1,
            'train_loss': avg_loss,
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
        })
        
        # Save checkpoint
        checkpoint_path = os.path.join(models_folder, f'encoder_decoder_4digit_epoch_{epoch + 1}.pth')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_accuracy': train_acc,
            'val_accuracy': val_acc,
        }, checkpoint_path)
        
        wandb_logger.save_artifact(checkpoint_path, f'model_epoch_{epoch + 1}')

    print(f'Training completed. Final Validation Accuracy: {val_acc:.2f}%')
    wandb_logger.finish()
    
    # Return components needed for final evaluation
    return model, device, test_loader

def test_encoder_decoder(model, device, test_loader):
    """Final comprehensive evaluation on test dataset"""
    print("\n" + "="*50)
    print("FINAL EVALUATION ON TEST DATASET")
    print("="*50)
    
    model.eval()
    final_correct = 0
    final_total = 0
    digit_correct = [0] * 4  # Track accuracy per digit position
    digit_total = [0] * 4
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Final Evaluation"):
            img = batch['image'].unsqueeze(1).to(device)
            labels = batch['labels'].to(device)
            
            # Generate sequences
            generated = model.generate(img)
            
            # Overall sequence accuracy
            sequence_correct = (generated == labels).all(dim=1)
            final_correct += sequence_correct.sum().item()
            final_total += img.size(0)
            
            # Per-digit accuracy
            for i in range(4):
                digit_correct[i] += (generated[:, i] == labels[:, i]).sum().item()
                digit_total[i] += img.size(0)
    
    final_accuracy = 100 * final_correct / final_total
    print(f"Final Test Accuracy (Complete Sequences): {final_accuracy:.2f}%")
    print(f"Correct Sequences: {final_correct}/{final_total}")
    
    # Print per-digit accuracies  
    for i in range(4):
        digit_acc = 100 * digit_correct[i] / digit_total[i]
        print(f"Digit {i+1} Accuracy: {digit_acc:.2f}%")
    
    # Initialize wandb logger for final metrics
    config = {
        'evaluation': 'final_test',
        'architecture': 'encoder_decoder_4digit'
    }
    
    run_name = os.getenv('WANDB_RUN_NAME_ENCODER_DECODER_TEST', 'vit-mnist-encoder-decoder-test')
    wandb_logger = WandbLogger(config, run_name=run_name)
    
    # Log final metrics
    wandb_logger.log_metrics({
        'final_test_accuracy': final_accuracy,
        'final_digit_1_accuracy': 100 * digit_correct[0] / digit_total[0],
        'final_digit_2_accuracy': 100 * digit_correct[1] / digit_total[1],
        'final_digit_3_accuracy': 100 * digit_correct[2] / digit_total[2],
        'final_digit_4_accuracy': 100 * digit_correct[3] / digit_total[3],
    })
    
    wandb_logger.finish()
    return final_accuracy

if __name__ == "__main__":
    train_loader, test_loader = get_4digit_mnist_loaders()
    model, device, test_loader = train_encoder_decoder(train_loader, test_loader)
    test_encoder_decoder(model, device, test_loader)
