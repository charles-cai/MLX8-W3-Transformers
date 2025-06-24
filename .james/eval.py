import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import json
from typing import Dict, List, Tuple

from data_processing import setup_data_loaders
from models.vit import VisionTransformer
from utils.wandb_utils import WandbLogger


def load_model(model_path: str, device: torch.device) -> VisionTransformer:
    """Load a trained model from checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model with default parameters (will be overridden by checkpoint)
    model = VisionTransformer(
        img_size=28,
        patch_size=7,
        in_channels=1,
        num_classes=10,
        embed_dim=64,
        depth=6,
        num_heads=8,
        mlp_ratio=4.0,
        dropout=0.1
    ).to(device)
    
    # Load state dict
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    return model


def evaluate_model(model: nn.Module, 
                  data_loader: DataLoader, 
                  device: torch.device,
                  return_predictions: bool = False) -> Dict:
    """Evaluate model and return metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    criterion = nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for data, target in tqdm(data_loader, desc="Evaluating"):
            # Handle patch format - reshape to original image format
            if len(data.shape) == 5:  # [B, patches, C, H, W]
                batch_size, num_patches, channels, patch_h, patch_w = data.shape
                # Reshape patches back to original 28x28 image
                data = data.view(batch_size, channels, 4, 4, patch_h, patch_w)
                data = data.permute(0, 1, 2, 4, 3, 5).contiguous()
                data = data.view(batch_size, channels, 28, 28)
            
            data, target = data.to(device), target.to(device)
            
            output = model(data)
            loss = criterion(output, target)
            
            total_loss += loss.item()
            probabilities = torch.softmax(output, dim=1)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
            all_predictions.extend(pred.squeeze().cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    accuracy = 100. * correct / total
    avg_loss = total_loss / len(data_loader)
    
    metrics = {
        'loss': avg_loss,
        'accuracy': accuracy,
        'correct': correct,
        'total': total
    }
    
    if return_predictions:
        return metrics, all_predictions, all_targets, all_probabilities
    else:
        return metrics


def plot_confusion_matrix(y_true: List[int], y_pred: List[int], 
                         class_names: List[str], save_path: str = None):
    """Plot confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_class_accuracy(y_true: List[int], y_pred: List[int], 
                       class_names: List[str], save_path: str = None):
    """Plot per-class accuracy."""
    cm = confusion_matrix(y_true, y_pred)
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(class_names)), class_accuracy)
    plt.xlabel('Class')
    plt.ylabel('Accuracy')
    plt.title('Per-Class Accuracy')
    plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Class accuracy plot saved to {save_path}")
    else:
        plt.show()
    plt.close()


def plot_sample_predictions(data_loader: DataLoader, 
                          model: nn.Module, 
                          device: torch.device,
                          num_samples: int = 16,
                          save_path: str = None):
    """Plot sample predictions with their true and predicted labels."""
    model.eval()
    
    # Get a batch of data
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Handle patch format - reshape to original image format
    if len(images.shape) == 5:  # [B, patches, C, H, W]
        batch_size, num_patches, channels, patch_h, patch_w = images.shape
        # Reshape patches back to original 28x28 image
        images = images.view(batch_size, channels, 4, 4, patch_h, patch_w)
        images = images.permute(0, 1, 2, 4, 3, 5).contiguous()
        images = images.view(batch_size, channels, 28, 28)
    
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images)
        probabilities = torch.softmax(outputs, dim=1)
        predictions = outputs.argmax(dim=1)
    
    # Convert to numpy
    images_np = images.cpu().numpy()
    labels_np = labels.cpu().numpy()
    predictions_np = predictions.cpu().numpy()
    probabilities_np = probabilities.cpu().numpy()
    
    # Create subplot
    fig, axes = plt.subplots(4, 4, figsize=(12, 12))
    axes = axes.ravel()
    
    for i in range(min(num_samples, len(images))):
        ax = axes[i]
        
        # Display image
        img = images_np[i, 0]  # Remove channel dimension
        ax.imshow(img, cmap='gray')
        
        # Add title with true and predicted labels
        true_label = labels_np[i]
        pred_label = predictions_np[i]
        confidence = probabilities_np[i, pred_label]
        
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}\nPred: {pred_label}\nConf: {confidence:.3f}', 
                    color=color, fontsize=10)
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Sample predictions saved to {save_path}")
    else:
        plt.show()
    plt.close()


def analyze_attention_weights(model: nn.Module, 
                            data_loader: DataLoader, 
                            device: torch.device,
                            num_samples: int = 5,
                            save_path: str = None):
    """Analyze attention weights from the model."""
    model.eval()
    
    # Hook to capture attention weights
    attention_weights = []
    
    def hook_fn(module, input, output):
        attention_weights.append(output.detach())
    
    # Register hooks on attention layers
    hooks = []
    for block in model.blocks:
        hooks.append(block.attn.register_forward_hook(hook_fn))
    
    # Get a few samples
    data_iter = iter(data_loader)
    images, labels = next(data_iter)
    
    # Handle patch format - reshape to original image format
    if len(images.shape) == 5:  # [B, patches, C, H, W]
        batch_size, num_patches, channels, patch_h, patch_w = images.shape
        # Reshape patches back to original 28x28 image
        images = images.view(batch_size, channels, 4, 4, patch_h, patch_w)
        images = images.permute(0, 1, 2, 4, 3, 5).contiguous()
        images = images.view(batch_size, channels, 28, 28)
    
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = model(images[:num_samples])
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    if attention_weights:
        # Plot attention weights for first sample
        attn = attention_weights[0][0]  # First layer, first sample
        attn_mean = attn.mean(dim=0).cpu().numpy()  # Average over heads
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_mean, cmap='viridis', cbar_kws={'label': 'Attention Weight'})
        plt.title('Attention Weights Heatmap (First Layer)')
        plt.xlabel('Key Position')
        plt.ylabel('Query Position')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Attention weights saved to {save_path}")
        else:
            plt.show()
        plt.close()


def save_evaluation_results(metrics: Dict, 
                          predictions: List[int], 
                          targets: List[int],
                          probabilities: List[List[float]],
                          save_path: str):
    """Save evaluation results to JSON file."""
    # Convert NumPy types to Python native types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj
    
    # Convert all data to JSON-serializable types
    metrics_converted = convert_numpy_types(metrics)
    predictions_converted = convert_numpy_types(predictions)
    targets_converted = convert_numpy_types(targets)
    probabilities_converted = convert_numpy_types(probabilities)
    
    results = {
        'metrics': metrics_converted,
        'predictions': predictions_converted,
        'targets': targets_converted,
        'probabilities': probabilities_converted,
        'classification_report': convert_numpy_types(classification_report(targets, predictions, output_dict=True))
    }
    
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation results saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Vision Transformer on MNIST')
    parser.add_argument('--model-path', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--batch-size', type=int, default=32, help='Batch size')
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--output-dir', type=str, default='eval_results', help='Output directory')
    parser.add_argument('--plot-attention', action='store_true', help='Plot attention weights')
    parser.add_argument('--wandb', action='store_true', help='Log results to W&B')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"Loading model from {args.model_path}")
    model = load_model(args.model_path, device)
    
    # Data loaders
    print("Setting up data loaders...")
    train_loader, val_loader = setup_data_loaders(batch_size=args.batch_size)
    
    # Evaluate on validation set
    print("Evaluating model...")
    metrics, predictions, targets, probabilities = evaluate_model(
        model, val_loader, device, return_predictions=True
    )
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Loss: {metrics['loss']:.4f}")
    print(f"Accuracy: {metrics['accuracy']:.2f}%")
    print(f"Correct: {metrics['correct']}/{metrics['total']}")
    print("\nClassification Report:")
    print(classification_report(targets, predictions))
    
    # Create plots
    class_names = [str(i) for i in range(10)]
    
    # Confusion matrix
    cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(targets, predictions, class_names, cm_path)
    
    # Per-class accuracy
    acc_path = os.path.join(args.output_dir, 'class_accuracy.png')
    plot_class_accuracy(targets, predictions, class_names, acc_path)
    
    # Sample predictions
    samples_path = os.path.join(args.output_dir, 'sample_predictions.png')
    plot_sample_predictions(val_loader, model, device, save_path=samples_path)
    
    # Attention weights (if requested)
    if args.plot_attention:
        attn_path = os.path.join(args.output_dir, 'attention_weights.png')
        analyze_attention_weights(model, val_loader, device, save_path=attn_path)
    
    # Save results
    results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    save_evaluation_results(metrics, predictions, targets, probabilities, results_path)
    
    # W&B logging
    if args.wandb:
        logger = WandbLogger("vision-transformer-mnist-eval")
        logger.log_metrics(metrics)
        logger.log_confusion_matrix(
            torch.tensor(targets), 
            torch.tensor(predictions), 
            class_names
        )
        logger.finish()
    
    print(f"\nAll results saved to {args.output_dir}/")


if __name__ == '__main__':
    main() 