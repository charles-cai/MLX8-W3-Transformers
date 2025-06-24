import wandb
import torch
import torch.nn as nn
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader
import os


class WandbLogger:
    """Utility class for Weights & Biases logging."""
    
    def __init__(self, project_name: str, run_name: Optional[str] = None, config: Optional[Dict] = None):
        """
        Initialize W&B logger.
        
        Args:
            project_name: Name of the W&B project
            run_name: Name of this specific run
            config: Configuration dictionary to log
        """
        self.project_name = project_name
        self.run_name = run_name
        self.config = config or {}
        
        # Initialize wandb
        wandb.init(
            project=project_name,
            name=run_name,
            config=config,
            reinit=True
        )
        
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics to W&B."""
        if step is not None:
            wandb.log(metrics, step=step)
        else:
            wandb.log(metrics)
    
    def log_model(self, model: nn.Module, model_name: str = "model"):
        """Log model architecture and parameters."""
        # Log model architecture
        wandb.watch(model, log="all", log_freq=100)
        
        # Save model file
        torch.save(model.state_dict(), f"{model_name}.pth")
        wandb.save(f"{model_name}.pth")
    
    def log_images(self, images: torch.Tensor, labels: Optional[torch.Tensor] = None, 
                   caption: str = "Training Images", max_images: int = 16):
        """Log images to W&B."""
        if images.dim() == 4:
            # Convert to numpy and normalize to [0, 1]
            images_np = images.detach().cpu().numpy()
            if images_np.max() > 1.0:
                images_np = images_np / 255.0
            
            # Take first batch and limit number of images
            images_np = images_np[:max_images]
            
            # Create captions if labels provided
            captions = None
            if labels is not None:
                captions = [f"Label: {label.item()}" for label in labels[:max_images]]
            
            wandb.log({caption: [wandb.Image(img, caption=cap) for img, cap in zip(images_np, captions or [None] * len(images_np))]})
    
    def log_confusion_matrix(self, y_true: torch.Tensor, y_pred: torch.Tensor, 
                           class_names: Optional[list] = None):
        """Log confusion matrix."""
        y_true_np = y_true.detach().cpu().numpy()
        y_pred_np = y_pred.detach().cpu().numpy()
        
        wandb.log({"confusion_matrix": wandb.plot.confusion_matrix(
            probs=None,
            y_true=y_true_np,
            preds=y_pred_np,
            class_names=class_names
        )})
    
    def log_learning_rate(self, optimizer: torch.optim.Optimizer, step: Optional[int] = None):
        """Log learning rate."""
        lr = optimizer.param_groups[0]['lr']
        self.log_metrics({"learning_rate": lr}, step)
    
    def log_gradients(self, model: nn.Module, step: Optional[int] = None):
        """Log gradient norms."""
        total_norm = 0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count > 0:
            total_norm = total_norm ** (1. / 2)
            self.log_metrics({"gradient_norm": total_norm}, step)
    
    def log_attention_weights(self, attention_weights: torch.Tensor, step: Optional[int] = None):
        """Log attention weights as heatmap."""
        # attention_weights shape: (batch_size, num_heads, seq_len, seq_len)
        if attention_weights.dim() == 4:
            # Take mean over heads and first batch
            attn_mean = attention_weights[0].mean(dim=0).detach().cpu().numpy()
            
            # Create heatmap
            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(attn_mean, cmap='viridis')
            ax.set_title('Attention Weights Heatmap')
            ax.set_xlabel('Key Position')
            ax.set_ylabel('Query Position')
            plt.colorbar(im)
            
            wandb.log({"attention_heatmap": wandb.Image(fig)}, step=step)
            plt.close(fig)
    
    def log_patch_embeddings(self, patch_embeddings: torch.Tensor, step: Optional[int] = None):
        """Log patch embeddings visualization."""
        # patch_embeddings shape: (batch_size, num_patches, embed_dim)
        if patch_embeddings.dim() == 3:
            # Take first batch and reduce dimensionality for visualization
            patches = patch_embeddings[0].detach().cpu().numpy()
            
            # Use PCA or t-SNE for dimensionality reduction
            from sklearn.decomposition import PCA
            pca = PCA(n_components=2)
            patches_2d = pca.fit_transform(patches)
            
            # Create scatter plot
            fig, ax = plt.subplots(figsize=(8, 6))
            scatter = ax.scatter(patches_2d[:, 0], patches_2d[:, 1], c=range(len(patches_2d)), cmap='viridis')
            ax.set_title('Patch Embeddings (PCA)')
            ax.set_xlabel('PCA Component 1')
            ax.set_ylabel('PCA Component 2')
            plt.colorbar(scatter)
            
            wandb.log({"patch_embeddings": wandb.Image(fig)}, step=step)
            plt.close(fig)
    
    def finish(self):
        """Finish the W&B run."""
        wandb.finish()


def setup_wandb(config: Dict[str, Any], run_name: Optional[str] = None) -> WandbLogger:
    """Setup W&B logger with configuration."""
    return WandbLogger(
        project_name="vision-transformer-mnist",
        run_name=run_name,
        config=config
    )


def log_training_step(logger: WandbLogger, 
                     loss: float, 
                     accuracy: float, 
                     step: int,
                     learning_rate: float,
                     images: Optional[torch.Tensor] = None,
                     labels: Optional[torch.Tensor] = None):
    """Log a single training step."""
    metrics = {
        "train/loss": loss,
        "train/accuracy": accuracy,
        "train/learning_rate": learning_rate
    }
    
    logger.log_metrics(metrics, step=step)
    
    # Log images occasionally
    if step % 100 == 0 and images is not None:
        logger.log_images(images, labels, caption="Training Images")


def log_validation_step(logger: WandbLogger,
                       loss: float,
                       accuracy: float,
                       step: int,
                       images: Optional[torch.Tensor] = None,
                       labels: Optional[torch.Tensor] = None,
                       predictions: Optional[torch.Tensor] = None):
    """Log a single validation step."""
    metrics = {
        "val/loss": loss,
        "val/accuracy": accuracy
    }
    
    logger.log_metrics(metrics, step=step)
    
    # Log images occasionally
    if step % 500 == 0 and images is not None:
        logger.log_images(images, labels, caption="Validation Images")
    
    # Log confusion matrix occasionally
    if step % 1000 == 0 and predictions is not None and labels is not None:
        logger.log_confusion_matrix(labels, predictions)


def create_wandb_config(model_config: Dict, training_config: Dict) -> Dict:
    """Create a comprehensive configuration dictionary for W&B."""
    config = {
        "model": model_config,
        "training": training_config,
        "dataset": {
            "name": "MNIST",
            "img_size": 28,
            "num_classes": 10,
            "patch_size": 7
        }
    }
    return config 