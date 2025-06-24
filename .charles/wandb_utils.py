import wandb
import os
from dotenv import load_dotenv
import torch
import yaml

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

class WandbSweeper:
    """Handle wandb sweep creation and execution"""
    
    def __init__(self):
        load_dotenv()
        self.api_key = os.getenv('WANDB_API_KEY')
        self.project = os.getenv('WANDB_PROJECT', 'mlx8-week3-transformers')
        
        if self.api_key:
            wandb.login(key=self.api_key)
    
    def create_sweep(self, config_file):
        """Create a wandb sweep from config file"""
        with open(config_file, 'r') as f:
            sweep_config = yaml.safe_load(f)
        
        sweep_id = wandb.sweep(sweep_config, project=self.project)
        print(f"Created sweep: {sweep_id}")
        print(f"Sweep URL: https://wandb.ai/{wandb.api.default_entity}/{self.project}/sweeps/{sweep_id}")
        return sweep_id
    
    def run_agent(self, sweep_id, count=None):
        """Run wandb agent for a sweep"""
        sweep_count = count or int(os.getenv('WANDB_SWEEP_COUNT', '10'))
        wandb.agent(sweep_id, count=sweep_count)
    
    def create_and_run(self, config_file, count=None):
        """Create sweep and run agent"""
        sweep_id = self.create_sweep(config_file)
        self.run_agent(sweep_id, count)
        return sweep_id
        return sweep_id

def main():
    """Main function for running sweeps directly"""
    parser = argparse.ArgumentParser(description='Run wandb hyperparameter sweep')
    parser.add_argument('--model', choices=['encoder_only', 'encoder_decoder'], 
                       required=True, help='Model type to sweep')
    parser.add_argument('--count', type=int, default=None, 
                       help='Number of sweep runs (default from env)')
    parser.add_argument('--create-only', action='store_true',
                       help='Only create sweep, don\'t run agent')
    
    args = parser.parse_args()
    
    # Select config file
    config_files = {
        'encoder_only': 'sweep_config_encoder_only.yaml',
        'encoder_decoder': 'sweep_config_encoder_decoder.yaml'
    }
    
    config_file = config_files[args.model]
    
    if not os.path.exists(config_file):
        print(f"Error: Config file {config_file} not found")
        return
    
    sweeper = WandbSweeper()
    
    if args.create_only:
        sweep_id = sweeper.create_sweep(config_file)
        print(f"To run sweep agent: wandb agent {sweep_id}")
    else:
        sweep_id = sweeper.create_and_run(config_file, args.count)
        print(f"Completed sweep: {sweep_id}")

if __name__ == "__main__":
    main()
