#!/usr/bin/env python3
"""
Quick debug test to see phase credit in action with detailed output
"""

import sys
import os
sys.path.insert(0, '/home/Karan/crazy_projects/on-policy')

def run_debug_test():
    """Run a very short test with debug output"""
    import subprocess
    
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--scenario_name', 'simple_spread',
        '--algorithm_name', 'mappo',
        '--experiment_name', 'debug_phase_credit',
        '--seed', '42',
        '--num_agents', '3',
        '--num_env_steps', '100000',  # Very short - ~125 episodes
        '--episode_length', '200',
        '--n_rollout_threads', '4',
        '--ppo_epoch', '3',
        '--num_mini_batch', '1',
        '--share_policy', 'False',  # Use separated runner
        # Phase credit settings
        '--use_phase_credit',
        '--credit_method', 'proxy',
        '--phase_method', 'fixed',
        '--phase_fixed_num', '2',  # Just 2 phases for simplicity
        '--credit_decay_lambda', '0.7',
        '--noop_action', '0',
        '--log_phase_credit',
        # Very frequent logging
        '--save_interval', '50000',
        '--log_interval', '20000',  # Log every ~25 episodes
        '--eval_interval', '200000',
        '--n_eval_rollout_threads', '1',
        '--eval_episodes', '1',
        '--user_name', 'debug_test',
        '--use_wandb', 'false'
    ]
    
    print("=" * 60)
    print("DEBUG TEST: Phase Credit with Detailed Output")
    print("=" * 60)
    print("Running ~125 episodes with phase credit enabled")
    print("Should see credit computation logs every ~25 episodes")
    print("Looking for: phase segmentation, credit weights, buffer updates")
    print("=" * 60)
    
    try:
        # Run with real-time output
        process = subprocess.Popen(cmd, 
                                 cwd='/home/Karan/crazy_projects/on-policy',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)
        
        # Print output in real-time
        for line in process.stdout:
            print(line.rstrip())
            
            # Look for specific phase credit indicators
            if 'phase_credit' in line.lower():
                print(">>> PHASE CREDIT LOG DETECTED <<<")
            elif 'credit' in line.lower() and ('weight' in line.lower() or 'segment' in line.lower()):
                print(">>> CREDIT-RELATED LOG <<<")
        
        process.wait()
        
        if process.returncode == 0:
            print("\n✅ Debug test completed!")
        else:
            print(f"\n❌ Debug test failed with code: {process.returncode}")
            
    except KeyboardInterrupt:
        print("\n⏹️ Debug test interrupted by user")
        process.terminate()
    except Exception as e:
        print(f"\n❌ Debug test failed: {e}")

if __name__ == "__main__":
    run_debug_test()
