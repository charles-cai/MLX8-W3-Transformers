#!/usr/bin/env python3
"""
Standalone script for running wandb hyperparameter sweeps.
Usage:
  # Regular training (unchanged)
  uv run encoder_only_models.py
  uv run encoder_decoder_models.py

  # Sweep training
  uv run wandb_run_sweep.py --model encoder_only --count 20
  uv run wandb_run_sweep.py --model encoder_decoder --create-only
"""

import sys
import os
import argparse
import wandb
import yaml
from dotenv import load_dotenv

# Import the training modules
try:
    import encoder_only_models
    import encoder_decoder_models
except ImportError as e:
    print(f"Warning: Could not import training modules: {e}")
    print("Make sure encoder_only_models.py and encoder_decoder_models.py are in the same directory")

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
    
    def create_train_function(self, model_type):
        """Create training function for wandb agent"""
        def train():
            # Initialize wandb run
            wandb.init()
            
            # Get hyperparameters from wandb config
            config = wandb.config
            
            # Convert config to args-like object or dictionary
            args_dict = dict(config)
            
            try:
                if model_type == 'encoder_only':
                    # Call encoder_only_models main with sweep parameters
                    encoder_only_models.main(args_dict)
                elif model_type == 'encoder_decoder':
                    # Call encoder_decoder_models main with sweep parameters
                    encoder_decoder_models.main(args_dict)
                
                print("Training completed successfully")
                wandb.finish()
                
            except Exception as e:
                print(f"Training failed with error: {e}")
                wandb.finish(exit_code=1)
                raise
        
        return train
    
    def run_agent(self, sweep_id, model_type, count=None):
        """Run wandb agent for a sweep"""
        sweep_count = count or int(os.getenv('WANDB_SWEEP_COUNT', '10'))
        train_function = self.create_train_function(model_type)
        wandb.agent(sweep_id, function=train_function, count=sweep_count)
    
    def create_and_run(self, config_file, model_type, count=None):
        """Create sweep and run agent"""
        sweep_id = self.create_sweep(config_file)
        self.run_agent(sweep_id, model_type, count)
        return sweep_id

def main():
    parser = argparse.ArgumentParser(
        description='Run wandb hyperparameter sweep for transformer models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run encoder-only sweep with 20 trials
  uv run wandb_run_sweep.py --model encoder_only --count 20
  
  # Run encoder-decoder sweep with default count
  uv run wandb_run_sweep.py --model encoder_decoder
  
  # Create sweep only (don't run agent)
  uv run wandb_run_sweep.py --model encoder_only --create-only
        """
    )
    
    parser.add_argument('--model', choices=['encoder_only', 'encoder_decoder'], 
                       required=True, help='Model type to sweep')
    parser.add_argument('--count', type=int, default=None, 
                       help='Number of sweep runs (default from WANDB_SWEEP_COUNT env var)')
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
        sweep_id = sweeper.create_and_run(config_file, args.model, args.count)
        print(f"Completed sweep: {sweep_id}")

if __name__ == "__main__":
    main()

