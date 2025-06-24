import torch
from torch.utils.data import random_split, DataLoader
from torch import reshape
from torchvision.datasets import MNIST
from torchvision.transforms import transforms


class MnistLoader():
    def __init__(self, datadir='.data/', batch_size=128):
        self.mnist_dataset = MNIST(root=datadir, download=True, train=True, transform=transforms.ToTensor())
        self.train_data, self.validation_data = random_split(self.mnist_dataset, [50000, 10000])
        self.train_loader = DataLoader(self.train_data, batch_size, shuffle=True)
        self.validation_loader = DataLoader(self.validation_data, batch_size, shuffle=False)
        print(f"mnist dataset loaded, train data size: {len(self.train_data)} validation data size: {len(self.validation_data)}")
        
    def get_loaders(self):
        return self.train_loader, self.validation_loader


def setup_data_loaders(batch_size=32):
    print("Loading MNIST data...")
    
    # Load MNIST data
    mnist_loader = MnistLoader(batch_size=batch_size)
    train_loader, val_loader = mnist_loader.get_loaders()
    
    # Return regular loaders (no patch splitting)
    # The Vision Transformer will handle patch embedding internally
    
    # Show data format
    sample_images, sample_labels = next(iter(train_loader))
    print(f"Image format: {sample_images.shape}")
    print(f"Labels format: {sample_labels.shape}")
    print(f"Vision Transformer will automatically split {sample_images.shape[2]}x{sample_images.shape[3]} images into patches")
    
    return train_loader, val_loader 

if __name__ == "__main__":
    setup_data_loaders(32)