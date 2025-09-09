#!/usr/bin/env python3

"""
Extended test script for phase credit system to see more phase transitions and credit variation.
This uses a more complex MPE scenario and longer episodes to trigger phase changes.
"""

import subprocess
import sys
import os

def run_extended_phase_credit_test():
    """Run an extended test to see phase transitions and credit variation."""
    
    print("=== Extended Phase Credit Test ===")
    print("Running MPE simple_adversary with longer episodes to see phase transitions...")
    print("This scenario has predator-prey dynamics which should create more varied phases\n")
    
    # Change to the project root directory
    os.chdir('/home/Karan/crazy_projects/on-policy')
    
    # Command for more complex scenario with longer episodes
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--algorithm_name', 'mappo', 
        '--experiment_name', 'phase_credit_extended_test',
        '--scenario_name', 'simple_adversary',  # More complex scenario
        '--num_agents', '3',
        '--num_adversaries', '1',
        '--seed', '1',
        '--n_training_threads', '1',
        '--n_rollout_threads', '4',  # Fewer parallel envs for clearer logs
        '--num_mini_batch', '1',
        '--episode_length', '50',  # Longer episodes for more phase changes
        '--num_env_steps', '4000',  # More steps
        '--ppo_epoch', '5',
        '--use_value_active_masks',
        '--use_eval',
        '--share_policy', 'False',  # IMPORTANT: Use separated runner
        # Phase credit specific arguments
        '--use_phase_credit', 'True',
        '--phase_credit_method', 'proxy',
        '--phase_min_length', '5',  # Longer minimum phase length
        '--phase_change_threshold', '0.2',  # More sensitive to changes
        '--credit_window_size', '8',
        '--phase_credit_log_interval', '1',  # Log every episode
        '--log_interval', '2',  # Less frequent general logs
        '--eval_interval', '1000',
        '--save_interval', '1000'
    ]
    
    print("Running command:")
    print(' '.join(cmd))
    print("\nLook for:")
    print("- '[PHASE_CREDIT] Episode X:' messages with varying credit stats")
    print("- Different mean and std values across episodes")
    print("- 'Phase X detected' messages showing phase transitions")
    print("="*60)
    
    try:
        # Run the command
        result = subprocess.run(cmd, cwd='/home/Karan/crazy_projects/on-policy', 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n" + "="*60)
            print("✅ Extended test completed successfully!")
            print("Check the logs above for phase credit statistics and phase transitions.")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Error running test: {e}")

if __name__ == "__main__":
    run_extended_phase_credit_test()
