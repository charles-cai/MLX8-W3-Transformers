import wandb
import os
from dotenv import load_dotenv
import torch

class WandbLogger:
    def __init__(self, config=None, run_name=None):
        # Load environment variables
        load_dotenv()
        
        self._WANDB_API_KEY = os.getenv('WANDB_API_KEY')
        self._WANDB_PROJECT = os.getenv('WANDB_PROJECT', 'mlx8-week3-transformers')
        self._WANDB_RUN_NAME = run_name or os.getenv('WANDB_RUN_NAME', 'vit-mnist-encoder-only')
        self._WANDB_ENTITY = os.getenv('WANDB_ENTITY')
        
        # Initialize wandb
        if self._WANDB_API_KEY:
            try:
                wandb.login(key=self._WANDB_API_KEY)
            except Exception as e:
                print(f"Warning: Failed to login to wandb: {e}")
        
        # Initialize run with error handling
        try:
            # Try with entity first
            if self._WANDB_ENTITY:
                self.run = wandb.init(
                    project=self._WANDB_PROJECT,
                    name=self._WANDB_RUN_NAME,
                    entity=self._WANDB_ENTITY,
                    config=config,
                    mode="online"
                )
            else:
                # Fallback without entity
                self.run = wandb.init(
                    project=self._WANDB_PROJECT,
                    name=self._WANDB_RUN_NAME,
                    config=config,
                    mode="online"
                )
        except Exception as e:
            print(f"Warning: Failed to initialize wandb: {e}")
            print("Running in offline mode or without wandb logging")
            self.run = None
    
    def log_metrics(self, metrics, step=None):
        """Log metrics to wandb"""
        if self.run:
            wandb.log(metrics, step=step)
    
    def log_model(self, model, model_name="model"):
        """Log model to wandb"""
        if self.run:
            wandb.watch(model, log="all")
    
    def log_hyperparameters(self, config):
        """Log hyperparameters to wandb"""
        if self.run:
            wandb.config.update(config)
    
    def finish(self):
        """Finish wandb run"""
        if self.run:
            wandb.finish()
    
    def save_artifact(self, file_path, artifact_name, artifact_type="model"):
        """Save file as wandb artifact"""
        if self.run:
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            artifact.add_file(file_path)
            wandb.log_artifact(artifact)
