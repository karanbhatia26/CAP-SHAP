#!/usr/bin/env python3
"""
Test phase credit with settings that should show more variation
"""

import sys
import os
sys.path.insert(0, '/home/Karan/crazy_projects/on-policy')

def run_varied_credit_test():
    """Run test with settings to show credit variation"""
    import subprocess
    
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--scenario_name', 'simple_spread',
        '--algorithm_name', 'mappo',
        '--experiment_name', 'varied_credit_test',
        '--seed', '123',
        '--num_agents', '3',
        '--num_env_steps', '150000',  # ~190 episodes
        '--episode_length', '200',
        '--n_rollout_threads', '4',
        '--ppo_epoch', '3',
        '--num_mini_batch', '1',
        '--share_policy', 'False',
        # Phase credit with more variation
        '--use_phase_credit',
        '--credit_method', 'proxy',
        '--phase_method', 'fixed',
        '--phase_fixed_num', '4',  # More phases for variation
        '--credit_decay_lambda', '0.3',  # Stronger temporal bias
        '--noop_action', '4',  # Different noop action
        '--log_phase_credit',
        # Fast logging
        '--save_interval', '75000',
        '--log_interval', '30000',
        '--eval_interval', '300000',
        '--n_eval_rollout_threads', '1',
        '--eval_episodes', '1',
        '--user_name', 'varied_test',
        '--use_wandb', 'false'
    ]
    
    print("=" * 60)
    print("VARIED CREDIT TEST")
    print("=" * 60)
    print("Settings for more variation:")
    print("- 4 phases instead of 2")
    print("- Different noop_action (4 instead of 0)")
    print("- Stronger temporal decay (0.3 instead of 0.7)")
    print("=" * 60)
    
    try:
        process = subprocess.Popen(cmd, 
                                 cwd='/home/Karan/crazy_projects/on-policy',
                                 stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT,
                                 universal_newlines=True,
                                 bufsize=1)
        
        episode_count = 0
        for line in process.stdout:
            print(line.rstrip())
            
            if '[PHASE_CREDIT]' in line and 'Episode' in line:
                episode_count += 1
                if 'mean=' in line and 'std=' in line:
                    # Extract credit stats
                    if 'std=0.0000' not in line:  # Look for non-uniform credits
                        print("🎯 NON-UNIFORM CREDIT DETECTED!")
        
        process.wait()
        print(f"\n✅ Varied credit test completed! Logged {episode_count} episodes with credit stats.")
        
    except KeyboardInterrupt:
        print("\n⏹️ Test interrupted")
        process.terminate()
    except Exception as e:
        print(f"\n❌ Test failed: {e}")

if __name__ == "__main__":
    run_varied_credit_test()
