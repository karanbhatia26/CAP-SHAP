#!/usr/bin/env python3

"""
Phase credit test with simple_tag scenario which has interesting chase dynamics.
"""

import subprocess
import sys
import os

def run_phase_credit_tag_test():
    """Run phase credit test with simple_tag scenario."""
    
    print("=== Phase Credit Test - Simple Tag ===")
    print("Running MPE simple_tag with phase credit...")
    print("This scenario has chase dynamics which should create phase transitions\n")
    
    # Change to the project root directory
    os.chdir('/home/Karan/crazy_projects/on-policy')
    
    # Command for simple_tag scenario
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--algorithm_name', 'mappo', 
        '--experiment_name', 'phase_credit_tag_test',
        '--scenario_name', 'simple_tag',  # Chase scenario
        '--num_agents', '4',
        '--num_adversaries', '1',
        '--seed', '1',
        '--n_training_threads', '1',
        '--n_rollout_threads', '4',  # Fewer parallel envs for clearer logs
        '--num_mini_batch', '1',
        '--episode_length', '40',  # Longer episodes
        '--num_env_steps', '3000',  
        '--ppo_epoch', '5',
        '--use_value_active_masks',
        '--use_eval',
        '--share_policy', 'False',  # IMPORTANT: Use separated runner
        # Phase credit specific arguments
        '--use_phase_credit', 'True',
        '--phase_credit_method', 'proxy',
        '--phase_min_length', '4',
        '--phase_change_threshold', '0.25',  
        '--credit_window_size', '6',
        '--phase_credit_log_interval', '1',  # Log every episode
        '--log_interval', '3',  
        '--eval_interval', '1000',
        '--save_interval', '1000'
    ]
    
    print("Running command:")
    print(' '.join(cmd))
    print("\nLook for phase credit logs with different statistics across episodes...")
    print("="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd='/home/Karan/crazy_projects/on-policy', 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ Tag test completed successfully!")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")

if __name__ == "__main__":
    run_phase_credit_tag_test()
