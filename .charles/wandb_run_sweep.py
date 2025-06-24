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
sys.path.append(os.path.dirname(__file__))

from wandb_utils import WandbSweeper
import argparse

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
        sweep_id = sweeper.create_and_run(config_file, args.count)
        print(f"Completed sweep: {sweep_id}")

if __name__ == "__main__":
    main()
if __name__ == "__main__":
    main()
