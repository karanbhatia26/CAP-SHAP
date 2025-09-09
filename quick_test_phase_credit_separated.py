#!/usr/bin/env python3

"""
Quick test script for phase credit system in MPE using SEPARATED runner.
This script runs a short training session with phase credit enabled to verify the system works.
"""

import subprocess
import sys
import os

def run_phase_credit_test():
    """Run a quick test of the phase credit system with separated runner."""
    
    print("=== Quick Phase Credit Test (Separated Runner) ===")
    print("Running MPE simple_spread with phase credit enabled...")
    print("Using SEPARATED runner (--share_policy=False) which has phase credit support\n")
    
    # Change to the project root directory
    os.chdir('/home/Karan/crazy_projects/on-policy')
    
    # Command for separated runner with phase credit
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--algorithm_name', 'mappo', 
        '--experiment_name', 'phase_credit_test_separated',
        '--scenario_name', 'simple_spread',
        '--num_agents', '3',
        '--num_landmarks', '3',
        '--seed', '1',
        '--n_training_threads', '1',
        '--n_rollout_threads', '8',
        '--num_mini_batch', '1',
        '--episode_length', '25',
        '--num_env_steps', '2000',  # Very short for quick test
        '--ppo_epoch', '5',
        '--use_value_active_masks',
        '--use_eval',
        '--share_policy', 'False',  # IMPORTANT: Use separated runner
        # Phase credit specific arguments
        '--use_phase_credit', 'True',
        '--phase_credit_method', 'proxy',
        '--phase_min_length', '3',
        '--phase_change_threshold', '0.3',
        '--credit_window_size', '5',
        '--phase_credit_log_interval', '1',  # Log every episode
        '--log_interval', '1',
        '--eval_interval', '500',
        '--save_interval', '500'
    ]
    
    print("Running command:")
    print(' '.join(cmd))
    print("\nLook for these logs:")
    print("- 'Phase X detected' messages")
    print("- 'Phase Credit Stats' with mean/std values")
    print("- Per-agent credit values")
    print("="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd='/home/Karan/crazy_projects/on-policy', 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ Test completed successfully!")
            print("Phase credit system is working with separated runner.")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")

if __name__ == "__main__":
    run_phase_credit_test()
