import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.ndimage import zoom
from tqdm import tqdm
import os
from typing import Dict, List, Tuple, Optional
import pandas as pd

# Try to import UMAP and HDBSCAN, fallback to alternatives if not available
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    print("Warning: UMAP not available, will use PCA as fallback for dimensionality reduction")
    UMAP_AVAILABLE = False

try:
    import hdbscan
    HDBSCAN_AVAILABLE = True
except ImportError:
    print("Warning: HDBSCAN not available, will use KMeans as fallback for clustering")
    HDBSCAN_AVAILABLE = False

# Import model classes
from encoder_only_models import VitMNISTEncoder, get_mnist_data_loaders, PatchEmbed, LearnablePos, MultiHeadAttention, FeedForward, TransformerBlock
from encoder_decoder_models import EncoderDecoderModel, get_4digit_mnist_loaders, FourDigitPatchEmbed, FourDigitEncoder, FourDigitDecoder, TransformerEncoderBlock, TransformerDecoderBlock

# Add safe globals for PyTorch loading
torch.serialization.add_safe_globals([
    VitMNISTEncoder, PatchEmbed, LearnablePos, MultiHeadAttention, FeedForward, TransformerBlock,
    EncoderDecoderModel, FourDigitPatchEmbed, FourDigitEncoder, FourDigitDecoder, 
    TransformerEncoderBlock, TransformerDecoderBlock
])

class EmbeddingExtractor:
    """Extract patch embeddings from trained transformer models"""
    
    def __init__(self, model_path: str, model_type: str = 'encoder_only'):
        self.model_path = model_path
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.load_model()
    
    def load_model(self):
        """Load trained model from checkpoint"""
        try:
            # First try with weights_only=True (safer)
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=True)
        except Exception as e:
            print(f"Warning: weights_only=True failed ({e}), falling back to weights_only=False")
            # Fall back to weights_only=False for models with custom classes
            checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        
        if self.model_type == 'encoder_only':
            # Load encoder-only model with proper parameter extraction
            config = checkpoint.get('model_config', {})
            
            # Extract parameters safely, ensuring they are integers
            d_model = config.get('d_model', 128)
            n_heads = config.get('n_heads', 4)
            depth = config.get('depth', 4)
            patch_size = config.get('patch', 7)
            
            # Handle case where patch might be stored as 'patch_size' instead of 'patch'
            if 'patch_size' in config:
                patch_size = config.get('patch_size', 7)
            
            # Ensure all parameters are integers
            if not isinstance(d_model, int):
                d_model = 128
            if not isinstance(n_heads, int):
                n_heads = 4
            if not isinstance(depth, int):
                depth = 4
            if not isinstance(patch_size, int):
                patch_size = 7
                
            print(f"Loading encoder-only model with: d_model={d_model}, n_heads={n_heads}, depth={depth}, patch={patch_size}")
            
            self.model = VitMNISTEncoder(
                d_model=d_model,
                n_heads=n_heads,
                depth=depth,
                patch=patch_size,
                dropout=0.0  # No dropout for inference
            ).to(self.device)
        else:
            # Load encoder-decoder model - use default parameters if config not available
            config = checkpoint.get('model_config', {})
            
            # Extract parameters safely
            d_model = config.get('d_model', 256)
            n_heads = config.get('n_heads', 8)
            enc_depth = config.get('enc_depth', 6)
            dec_depth = config.get('dec_depth', 6)
            patch_size = config.get('patch_size', 7)
            
            # Ensure all parameters are integers
            if not isinstance(d_model, int):
                d_model = 256
            if not isinstance(n_heads, int):
                n_heads = 8
            if not isinstance(enc_depth, int):
                enc_depth = 6
            if not isinstance(dec_depth, int):
                dec_depth = 6
            if not isinstance(patch_size, int):
                patch_size = 7
                
            print(f"Loading encoder-decoder model with: d_model={d_model}, n_heads={n_heads}, enc_depth={enc_depth}, dec_depth={dec_depth}, patch_size={patch_size}")
            
            self.model = EncoderDecoderModel(
                d_model=d_model,
                n_heads=n_heads,
                enc_depth=enc_depth,
                dec_depth=dec_depth,
                dropout=0.0,  # No dropout for inference
                patch_size=patch_size
            ).to(self.device)
        
        # Load model state dict with error handling
        try:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        except KeyError:
            # Handle different checkpoint formats
            if 'state_dict' in checkpoint:
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                raise ValueError("No valid model state dict found in checkpoint")
        
        self.model.eval()
        print(f"Loaded {self.model_type} model from {self.model_path}")
    
    def extract_patch_embeddings(self, data_loader, num_samples: int = 1000) -> Dict:
        """Extract patch embeddings from model encoder"""
        embeddings = []
        labels = []
        images = []
        attention_weights = []
        
        with torch.no_grad():
            samples_collected = 0
            for batch in tqdm(data_loader, desc="Extracting embeddings"):
                if samples_collected >= num_samples:
                    break
                
                if self.model_type == 'encoder_only':
                    img = batch['image'].unsqueeze(1).to(self.device)
                    label = batch['label'].to(self.device)
                    
                    # Forward pass through patch embedding and positional encoding
                    B = img.size(0)
                    x = self.model.patch(img)  # (B, n_patches, d_model)
                    cls = self.model.cls_token.expand(B, -1, -1)
                    x = torch.cat([cls, x], dim=1)
                    x = self.model.pos(x)
                    
                    # Extract attention weights from first transformer block
                    attn_block = self.model.transformer_blocks[0]
                    attn_weights_batch = self.get_attention_weights(x, attn_block)
                    
                    # Store patch embeddings (excluding CLS token)
                    patch_embeds = x[:, 1:, :].cpu().numpy()  # (B, n_patches, d_model)
                    
                    for i in range(B):
                        if samples_collected >= num_samples:
                            break
                        embeddings.append(patch_embeds[i])  # (n_patches, d_model)
                        labels.append(label[i].cpu().item())
                        images.append(img[i, 0].cpu().numpy())
                        attention_weights.append(attn_weights_batch[i].cpu().numpy())
                        samples_collected += 1
                
                else:  # encoder_decoder
                    img = batch['image'].unsqueeze(1).to(self.device)
                    label = batch['labels'].to(self.device)
                    
                    # Get batch size
                    B = img.size(0)
                    
                    # Forward pass through encoder
                    x = self.model.encoder.patch_embed(img)  # (B, 64, d_model)
                    x = self.model.encoder.pos_embed(x)
                    
                    # Extract attention weights from first encoder block
                    attn_block = self.model.encoder.encoder_blocks[0]
                    attn_weights_batch = self.get_attention_weights(x, attn_block)
                    
                    patch_embeds = x.cpu().numpy()  # (B, 64, d_model)
                    
                    for i in range(B):
                        if samples_collected >= num_samples:
                            break
                        embeddings.append(patch_embeds[i])  # (64, d_model)
                        labels.append(label[i].cpu().numpy())  # 4 digits for encoder-decoder
                        images.append(img[i, 0].cpu().numpy())
                        attention_weights.append(attn_weights_batch[i].cpu().numpy())
                        samples_collected += 1
        
        return {
            'embeddings': embeddings,
            'labels': labels,
            'images': images,
            'attention_weights': attention_weights
        }
    
    def get_attention_weights(self, x, attention_block):
        """Extract attention weights from transformer block"""
        B, N, D = x.shape
        
        # Handle different attention block types
        if hasattr(attention_block, 'attention'):
            # Encoder-only model structure
            attention_layer = attention_block.attention
        elif hasattr(attention_block, 'self_attention'):
            # Encoder-decoder model structure (use self-attention)
            attention_layer = attention_block.self_attention
        else:
            raise ValueError("Unknown attention block structure")
        
        # Get Q, K, V from attention layer
        Q = attention_layer.w_q(x)
        K = attention_layer.w_k(x)
        
        # Reshape for multi-head attention
        n_heads = attention_layer.n_heads
        d_k = attention_layer.d_k
        
        Q = Q.view(B, N, n_heads, d_k).transpose(1, 2)
        K = K.view(B, N, n_heads, d_k).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (d_k ** 0.5)
        attn_weights = torch.softmax(scores, dim=-1)
        
        # Average across heads
        attn_weights = attn_weights.mean(dim=1)  # (B, N, N)
        
        return attn_weights

class EmbeddingVisualizer:
    """Visualize patch embeddings using dimensionality reduction"""
    
    def __init__(self, save_dir: str = '.data/visualizations'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
    
    def create_tsne_plot(self, embeddings_data: Dict, model_type: str):
        """Create t-SNE visualization of patch embeddings"""
        embeddings = embeddings_data['embeddings']
        labels = embeddings_data['labels']
        
        # Flatten embeddings: (n_samples, n_patches, d_model) -> (n_samples*n_patches, d_model)
        flat_embeddings = []
        flat_labels = []
        
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            for patch_idx in range(emb.shape[0]):
                flat_embeddings.append(emb[patch_idx])
                if model_type == 'encoder_only':
                    flat_labels.append(label)
                else:
                    # For encoder-decoder, use first digit as primary label
                    flat_labels.append(label[0] if isinstance(label, np.ndarray) else label)
        
        flat_embeddings = np.array(flat_embeddings)
        flat_labels = np.array(flat_labels)
        
        print(f"Running t-SNE on {flat_embeddings.shape[0]} patch embeddings...")
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        embeddings_2d = tsne.fit_transform(flat_embeddings[:5000])  # Limit for speed
        
        # Create plot
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                            c=flat_labels[:5000], cmap='tab10', alpha=0.6, s=1)
        plt.colorbar(scatter)
        plt.title(f'{model_type.title()} - t-SNE of Patch Embeddings')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        
        save_path = os.path.join(self.save_dir, f'{model_type}_tsne.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"t-SNE plot saved to {save_path}")
    
    def create_umap_plot(self, embeddings_data: Dict, model_type: str):
        """Create UMAP visualization with clustering (fallback to PCA + KMeans if UMAP/HDBSCAN unavailable)"""
        embeddings = embeddings_data['embeddings']
        labels = embeddings_data['labels']
        
        # Flatten embeddings
        flat_embeddings = []
        flat_labels = []
        
        for i, (emb, label) in enumerate(zip(embeddings, labels)):
            for patch_idx in range(emb.shape[0]):
                flat_embeddings.append(emb[patch_idx])
                if model_type == 'encoder_only':
                    flat_labels.append(label)
                else:
                    flat_labels.append(label[0] if isinstance(label, np.ndarray) else label)
        
        flat_embeddings = np.array(flat_embeddings)
        flat_labels = np.array(flat_labels)
        
        # Limit data for performance
        data_limit = 5000
        flat_embeddings = flat_embeddings[:data_limit]
        flat_labels = flat_labels[:data_limit]
        
        if UMAP_AVAILABLE:
            print(f"Running UMAP on {flat_embeddings.shape[0]} patch embeddings...")
            reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=15, min_dist=0.1)
            embeddings_2d = reducer.fit_transform(flat_embeddings)
            method_name = "UMAP"
        else:
            print(f"Running PCA on {flat_embeddings.shape[0]} patch embeddings...")
            reducer = PCA(n_components=2, random_state=42)
            embeddings_2d = reducer.fit_transform(flat_embeddings)
            method_name = "PCA"
        
        # Apply clustering
        if HDBSCAN_AVAILABLE:
            print("Running HDBSCAN clustering...")
            clusterer = hdbscan.HDBSCAN(min_cluster_size=50, min_samples=10)
            cluster_labels = clusterer.fit_predict(embeddings_2d)
            cluster_method = "HDBSCAN"
        else:
            print("Running KMeans clustering...")
            clusterer = KMeans(n_clusters=10, random_state=42, n_init=10)
            cluster_labels = clusterer.fit_predict(embeddings_2d)
            cluster_method = "KMeans"
        
        # Create subplot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Dimensionality reduction colored by digit labels
        scatter1 = ax1.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=flat_labels, cmap='tab10', alpha=0.6, s=1)
        ax1.set_title(f'{model_type.title()} - {method_name} by Digit Labels')
        ax1.set_xlabel(f'{method_name} 1')
        ax1.set_ylabel(f'{method_name} 2')
        plt.colorbar(scatter1, ax=ax1)
        
        # Dimensionality reduction colored by clusters
        scatter2 = ax2.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                              c=cluster_labels, cmap='viridis', alpha=0.6, s=1)
        ax2.set_title(f'{model_type.title()} - {method_name} with {cluster_method} Clusters')
        ax2.set_xlabel(f'{method_name} 1')
        ax2.set_ylabel(f'{method_name} 2')
        plt.colorbar(scatter2, ax=ax2)
        
        save_path = os.path.join(self.save_dir, f'{model_type}_{method_name.lower()}_{cluster_method.lower()}.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"{method_name} plot saved to {save_path}")
        
        return embeddings_2d, cluster_labels
    
    def visualize_attention_maps(self, embeddings_data: Dict, model_type: str, num_samples: int = 10):
        """Visualize attention maps for sample images"""
        images = embeddings_data['images'][:num_samples]
        attention_weights = embeddings_data['attention_weights'][:num_samples]
        labels = embeddings_data['labels'][:num_samples]
        
        if model_type == 'encoder_only':
            # For encoder-only: 28x28 images with 7x7 or 4x4 patches
            self._plot_single_digit_attention(images, attention_weights, labels)
        else:
            # For encoder-decoder: 56x56 stacked images with 8x8 patches
            self._plot_4digit_attention(images, attention_weights, labels)
    
    def _plot_single_digit_attention(self, images, attention_weights, labels):
        """Plot attention maps for single digit images"""
        fig, axes = plt.subplots(2, 10, figsize=(25, 6))
        
        for i in range(10):
            img = images[i]
            attn = attention_weights[i]
            label = labels[i]
            
            # Original image
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Digit: {label}')
            axes[0, i].axis('off')
            
            # Attention map (excluding CLS token)
            # Average attention from all tokens to all tokens, excluding CLS
            patch_attn = attn[1:, 1:].mean(axis=0)  # Average attention received by each patch
            
            # Reshape to spatial dimensions
            patch_size = int(np.sqrt(len(patch_attn)))
            attn_map = patch_attn.reshape(patch_size, patch_size)
            
            # Resize attention map to match image size using scipy
            attn_resized = zoom(attn_map, (28/patch_size, 28/patch_size))
            
            # Overlay attention on image
            axes[1, i].imshow(img, cmap='gray', alpha=0.7)
            axes[1, i].imshow(attn_resized, cmap='hot', alpha=0.5)
            axes[1, i].set_title(f'Attention Map')
            axes[1, i].axis('off')
        
        plt.suptitle('Single Digit Attention Maps')
        save_path = os.path.join(self.save_dir, 'encoder_only_attention_maps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Attention maps saved to {save_path}")
    
    def _plot_4digit_attention(self, images, attention_weights, labels):
        """Plot attention maps for 4-digit stacked images"""
        fig, axes = plt.subplots(2, 10, figsize=(25, 6))
        
        for i in range(10):
            img = images[i]  # 56x56
            attn = attention_weights[i]  # (64, 64) - 8x8 patches
            label = labels[i]  # 4 digits
            
            # Original image
            axes[0, i].imshow(img, cmap='gray')
            axes[0, i].set_title(f'Digits: {label}')
            axes[0, i].axis('off')
            
            # Attention map
            patch_attn = attn.mean(axis=0)  # Average attention received by each patch
            attn_map = patch_attn.reshape(8, 8)
            
            # Resize attention map to match image size using scipy
            attn_resized = zoom(attn_map, (56/8, 56/8))
            
            # Overlay attention on image
            axes[1, i].imshow(img, cmap='gray', alpha=0.7)
            axes[1, i].imshow(attn_resized, cmap='hot', alpha=0.5)
            axes[1, i].set_title(f'Attention Map')
            axes[1, i].axis('off')
        
        plt.suptitle('4-Digit Stacked Attention Maps')
        save_path = os.path.join(self.save_dir, 'encoder_decoder_attention_maps.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Attention maps saved to {save_path}")

def visualize_embeddings(model_path: str, model_type: str = 'encoder_only', num_samples: int = 1000):
    """Main function to extract and visualize embeddings"""
    print(f"Starting embedding visualization for {model_type} model...")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    # Initialize extractor
    extractor = EmbeddingExtractor(model_path, model_type)
    
    # Get appropriate data loader
    if model_type == 'encoder_only':
        _, test_loader = get_mnist_data_loaders(batch_size_test=64)
    else:
        _, test_loader = get_4digit_mnist_loaders(batch_size_test=64)
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings_data = extractor.extract_patch_embeddings(test_loader, num_samples)
    
    # Initialize visualizer
    visualizer = EmbeddingVisualizer()
    
    # Create visualizations
    print("Creating t-SNE visualization...")
    visualizer.create_tsne_plot(embeddings_data, model_type)
    
    print("Creating UMAP/PCA visualization...")
    visualizer.create_umap_plot(embeddings_data, model_type)
    
    print("Creating attention visualizations...")
    visualizer.visualize_attention_maps(embeddings_data, model_type)
    
    print("Visualization complete!")

# Usage examples
if __name__ == "__main__":
    # Check for environment variables first
    model_type = os.getenv('MODEL_TYPE', 'encoder_only')
    num_samples = int(os.getenv('NUM_SAMPLES', '500'))
    
    if model_type == 'encoder_only':
        model_path = os.getenv('MODEL_PATH', '.data/models/vit_mnist_epoch_8.pth')
    else:
        model_path = os.getenv('MODEL_PATH', '.data/models/encoder_decoder_4digit_epoch_8.pth')
    
    if os.path.exists(model_path):
        print(f"Using model: {model_path}")
        print(f"Model type: {model_type}")
        print(f"Number of samples: {num_samples}")
        visualize_embeddings(model_path, model_type, num_samples=num_samples)
    else:
        print(f"Model file not found: {model_path}")
        print("Available model files:")
        models_dir = '.data/models'
        if os.path.exists(models_dir):
            for file in os.listdir(models_dir):
                if file.endswith('.pth'):
                    print(f"  {os.path.join(models_dir, file)}")
        else:
            print(f"  Models directory not found: {models_dir}")
