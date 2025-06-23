from xml.parsers.expat import model
from datasets import load_dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import os
import glob

def get_mnist_data_loaders(batch_size_train=128, batch_size_test=256):
    # Debug: Check what files exist
    print("Files in .data/ylecun/mnist/:")
    if os.path.exists(".data/ylecun/mnist/"):
        for root, dirs, files in os.walk(".data/ylecun/mnist/"):
            for file in files:
                print(os.path.join(root, file))

    # Try with exact file paths
    train_files = glob.glob(".data/ylecun/mnist/**/train-*.parquet", recursive=True)
    test_files = glob.glob(".data/ylecun/mnist/**/test-*.parquet", recursive=True)

    print(f"Found train files: {train_files}")
    print(f"Found test files: {test_files}")

    # 1. Load from local directory (no download)
    train_ds = load_dataset("parquet", data_files=train_files)["train"]
    test_ds = load_dataset("parquet", data_files=test_files)["train"]

    train_ds = train_ds.with_format(
        "torch", 
        columns=["image", "label"],
        output_all_columns=False 
    )
    test_ds = test_ds.with_format(
        "torch", 
        columns=["image", "label"],
        output_all_columns=False
    )
    # 3. Build PyTorch loaders
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

class VitMNISTEncoder(nn.Module):
    """
    [PatchEmbed] → [Pos] → [Encoder×L] → optional [CLS] pooling → logits(10)
    """
    def __init__(self, d_model=128, n_heads=4, depth=4, patch=4):
        super().__init__()
        self.patch = PatchEmbed(patch, d_model)
        # Fix: Account for CLS token in positioning (49 patches + 1 CLS = 50 tokens)
        self.pos   = LearnablePos(50, d_model)
        enc_layer  = nn.TransformerEncoderLayer(
                        d_model, n_heads, 4*d_model, 0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)

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

        z = self.encoder(x)                      # (B,50,d_model)
        logits = self.head(z[:, 0])              # take [CLS] (index 0)
        return logits

def train(train_loader, test_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = VitMNISTEncoder().to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(5):  # Add epoch loop
        total_loss = 0
        for batch_idx, batch in enumerate(train_loader):
            # Fix: Handle batch structure from datasets
            img = batch['image'].float().unsqueeze(1) / 255.0  # Normalize and add channel dim
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
            img = batch['image'].float().unsqueeze(1) / 255.0
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