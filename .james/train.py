#!/usr/bin/env python3
"""
Simple training script for Vision Transformer on MNIST dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import os
import time
import wandb

from models.vit import VisionTransformer
from data_processing import setup_data_loaders
from utils.wandb_utils import setup_wandb, log_training_step, log_validation_step, create_wandb_config


def train_model(config=None):
    """Training function for Vision Transformer with optional wandb config."""
    
    # Use wandb config if provided, otherwise use defaults
    if config is None:
        config = {
            'batch_size': [16, 32, 64],
            'epochs': 1,
            'learning_rate': 0.001,
            'weight_decay': 0.01,
            'embed_dim': 64,
            'depth': 6,
            'num_heads': 8,
            'mlp_ratio': 4.0,
            'dropout': 0.1,
            'patch_size': 7,
            'use_cls_token': True
        }
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {DEVICE}")
    
    # Create model
    model = VisionTransformer(
        img_size=28,
        patch_size=config['patch_size'],
        in_channels=1,
        num_classes=10,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=config['dropout'],
        use_cls_token=config['use_cls_token']
    ).to(DEVICE)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup data loaders (now returns regular image loaders)
    train_loader, val_loader = setup_data_loaders(batch_size=config['batch_size'])
    
    # Setup optimizer and loss
    optimizer = optim.AdamW(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    criterion = nn.CrossEntropyLoss()
    
    # Training variables
    best_val_acc = 0.0
    global_step = 0
    
    print(f"Starting training for {config['epochs']} epochs...")
    print(f"Training batches: {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print("-" * 50)
    
    for epoch in range(config['epochs']):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        start_time = time.time()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            # Data is already in correct format: [B, C, H, W]
            data, target = data.to(DEVICE), target.to(DEVICE)
            
            # Forward pass
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Log to wandb if available
            if wandb.run is not None:
                batch_acc = 100. * pred.eq(target).sum().item() / target.size(0)
                wandb.log({
                    "train/batch_loss": loss.item(),
                    "train/batch_accuracy": batch_acc,
                    "train/learning_rate": config['learning_rate'],
                    "step": global_step
                })
                
                # Log images occasionally (fixed to handle batch properly)
                if batch_idx % 100 == 0:
                    # Take first 4 images from batch and log them individually
                    images_to_log = []
                    for i in range(min(4, data.shape[0])):
                        img = data[i].cpu()  # [C, H, W]
                        images_to_log.append(wandb.Image(img, caption=f"Image {i}, Label: {target[i].item()}"))
                    
                    wandb.log({
                        "train/images": images_to_log
                    })
            
            global_step += 1
            
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f'Epoch {epoch+1}/{config["epochs"]}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in val_loader:
                # Data is already in correct format: [B, C, H, W]
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                loss = criterion(output, target)
                
                val_loss += loss.item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                all_predictions.append(pred)
                all_targets.append(target)
        
        # Calculate metrics
        epoch_time = time.time() - start_time
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        
        # Log to wandb if available
        if wandb.run is not None:
            wandb.log({
                "epoch": epoch,
                "train/loss": avg_train_loss,
                "train/accuracy": train_acc,
                "val/loss": avg_val_loss,
                "val/accuracy": val_acc,
                "epoch_time": epoch_time
            })
            
            # Log confusion matrix occasionally
            if epoch % 2 == 0:
                all_preds = torch.cat(all_predictions)
                all_targets = torch.cat(all_targets)
                wandb.log({
                    "confusion_matrix": wandb.plot.confusion_matrix(
                        probs=None,
                        y_true=all_targets.cpu().numpy(),
                        preds=all_preds.cpu().numpy(),
                        class_names=[str(i) for i in range(10)]
                    )
                })
        
        print(f'Epoch {epoch+1}/{config["epochs"]}:')
        print(f'  Train - Loss: {avg_train_loss:.4f}, Acc: {train_acc:.2f}%')
        print(f'  Val   - Loss: {avg_val_loss:.4f}, Acc: {val_acc:.2f}%')
        print(f'  Time: {epoch_time:.2f}s')
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config
            }, 'best_model.pth')
            print(f'  ✓ New best model saved! Val Acc: {val_acc:.2f}%')
        
        print("-" * 50)
    
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return model, best_val_acc


def evaluate_model(model_path='best_model.pth'):
    """Simple evaluation function."""
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)
    config = checkpoint.get('config', {
        'embed_dim': 64, 'depth': 6, 'num_heads': 8, 'mlp_ratio': 4.0,
        'dropout': 0.1, 'patch_size': 7, 'use_cls_token': True
    })
    
    # Load model
    model = VisionTransformer(
        img_size=28, patch_size=config['patch_size'], in_channels=1, num_classes=10,
        embed_dim=config['embed_dim'], depth=config['depth'], num_heads=config['num_heads'], 
        mlp_ratio=config['mlp_ratio'], dropout=config['dropout'], use_cls_token=config['use_cls_token']
    ).to(DEVICE)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Get validation data
    _, val_loader = setup_data_loaders(batch_size=32)
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, target in val_loader:
            # Data is already in correct format: [B, C, H, W]
            data, target = data.to(DEVICE), target.to(DEVICE)
            output = model(data)
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
    
    accuracy = 100. * correct / total
    print(f'Test Accuracy: {accuracy:.2f}% ({correct}/{total})')
    return accuracy


def create_sweep_config():
    """Create wandb sweep configuration for hyperparameter optimization."""
    sweep_config = {
        'method': 'random',  # or 'grid', 'bayes'
        'name': 'vit-hyperparameter-sweep',
        'metric': {
            'name': 'val/accuracy',
            'goal': 'maximize'
        },
        'parameters': {
            'learning_rate': {
                'min': 0.0001,
                'max': 0.01,
                'distribution': 'log_uniform_values'
            },
            'batch_size': {
                'values': [16, 32, 64]
            },
            'embed_dim': {
                'values': [32, 64, 128]
            },
            'depth': {
                'values': [4, 6, 8]
            },
            'num_heads': {
                'values': [4, 8, 16]
            },
            'mlp_ratio': {
                'values': [2.0, 4.0, 8.0]
            },
            'dropout': {
                'min': 0.0,
                'max': 0.3,
                'distribution': 'uniform'
            },
            'weight_decay': {
                'min': 0.001,
                'max': 0.1,
                'distribution': 'log_uniform_values'
            },
            'epochs': {
                'value': 1
            },
            'patch_size': {
                'value': 7
            },
            'use_cls_token': {
                'value': True
            }
        }
    }
    return sweep_config


def sweep_agent():
    """Agent function for wandb sweep."""
    wandb.init()
    
    # Get the hyperparameters from wandb
    config = wandb.config
    
    # Convert wandb config to dictionary format
    config_dict = {
        'batch_size': config.batch_size,
        'epochs': config.epochs,
        'learning_rate': config.learning_rate,
        'weight_decay': config.weight_decay,
        'embed_dim': config.embed_dim,
        'depth': config.depth,
        'num_heads': config.num_heads,
        'mlp_ratio': config.mlp_ratio,
        'dropout': config.dropout,
        'patch_size': config.patch_size,
        'use_cls_token': config.use_cls_token
    }
    
    # Train the model
    model, best_acc = train_model(config_dict)
    
    # Log final results
    wandb.log({
        "best_val_accuracy": best_acc,
        "final_model": wandb.save("best_model.pth")
    })


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Simple ViT Training')
    parser.add_argument('--mode', choices=['train', 'eval', 'sweep'], default='train',
                       help='Mode: train, eval, or sweep')
    parser.add_argument('--model-path', default='best_model.pth',
                       help='Path to model for evaluation')
    parser.add_argument('--sweep-id', type=str, default=None,
                       help='Sweep ID to join (for distributed sweeps)')
    parser.add_argument('--project', type=str, default='vision-transformer-mnist',
                       help='Wandb project name')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        model, best_acc = train_model()
        print(f"\nTraining finished! Best accuracy: {best_acc:.2f}%")
    elif args.mode == 'eval':
        accuracy = evaluate_model(args.model_path)
        print(f"Evaluation finished! Accuracy: {accuracy:.2f}%")
    elif args.mode == 'sweep':
        if args.sweep_id:
            # Join existing sweep
            wandb.agent(args.sweep_id, function=sweep_agent, project=args.project)
        else:
            # Create new sweep
            sweep_config = create_sweep_config()
            sweep_id = wandb.sweep(sweep_config, project=args.project)
            print(f"Created sweep with ID: {sweep_id}")
            print("Run the following command to start the sweep agent:")
            print(f"python train.py --mode sweep --sweep-id {sweep_id}")
