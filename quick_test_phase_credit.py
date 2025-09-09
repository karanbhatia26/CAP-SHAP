#!/usr/bin/env python3
"""
Quick test script for phase credit with MPE environment - 2000-3000 episodes only
"""

import sys
import os
sys.path.insert(0, '/home/Karan/crazy_projects/on-policy')

def run_quick_mpe_test():
    """Run MPE with phase credit for 2000-3000 episodes"""
    import subprocess
    
    cmd = [
        'python', '-m', 'onpolicy.scripts.train.train_mpe',
        '--env_name', 'MPE',
        '--scenario_name', 'simple_spread',
        '--algorithm_name', 'mappo',
        '--experiment_name', 'quick_phase_credit_test',
        '--seed', '1',
        '--num_agents', '3',
        '--num_env_steps', '2500000',  # ~2500 episodes with 200 step episodes and 8 threads
        '--episode_length', '200',
        '--n_rollout_threads', '8',
        '--ppo_epoch', '10',
        '--num_mini_batch', '1',
        # Phase credit settings
        '--use_phase_credit',
        '--credit_method', 'proxy',
        '--phase_method', 'fixed',
        '--phase_fixed_num', '3',
        '--credit_decay_lambda', '0.5',
        '--noop_action', '0',
        '--log_phase_credit',
        # Fast testing settings
        '--save_interval', '500000',
        '--log_interval', '100000',
        '--eval_interval', '1000000',
        '--n_eval_rollout_threads', '1',
        '--eval_episodes', '5',
        '--user_name', 'test_user',
        '--use_wandb', 'false'
    ]
    
    print("=" * 60)
    print("RUNNING QUICK MPE PHASE CREDIT TEST")
    print("=" * 60)
    print(f"Command: {' '.join(cmd)}")
    print("Expected: ~2500 episodes in MPE simple_spread")
    print("Features: Phase credit with proxy Shapley method")
    print("Duration: Should complete in 10-15 minutes")
    print("=" * 60)
    
    try:
        result = subprocess.run(cmd, cwd='/home/Karan/crazy_projects/on-policy', 
                              capture_output=False, text=True)
        
        if result.returncode == 0:
            print("\n✅ QUICK TEST COMPLETED SUCCESSFULLY!")
            print("Check TensorBoard logs for phase credit metrics:")
            print("  tensorboard --logdir onpolicy/scripts/results/MPE/simple_spread/mappo/quick_phase_credit_test")
        else:
            print(f"\n❌ Test failed with return code: {result.returncode}")
            
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")

def run_quick_proxy_vs_deep_comparison():
    """Run both proxy and deep methods for comparison"""
    print("=" * 60)
    print("QUICK COMPARISON: PROXY vs DEEP SHAPLEY")
    print("=" * 60)
    
    configs = [
        {
            'method': 'proxy',
            'exp_name': 'quick_proxy_test',
            'extra_args': []
        },
        {
            'method': 'deep', 
            'exp_name': 'quick_deep_test',
            'extra_args': [
                '--deep_shapley_lr', '1e-3',
                '--deep_shapley_hidden', '32',  # Smaller for speed
                '--deep_shapley_buffer', '500',
                '--deep_shapley_coalitions', '4'
            ]
        }
    ]
    
    for config in configs:
        print(f"\nRunning {config['method']} Shapley test...")
        
        cmd = [
            'python', '-m', 'onpolicy.scripts.train.train_mpe',
            '--env_name', 'MPE',
            '--scenario_name', 'simple_spread',
            '--algorithm_name', 'mappo',
            '--experiment_name', config['exp_name'],
            '--seed', '1',
            '--num_agents', '3',
            '--num_env_steps', '1600000',  # ~2000 episodes
            '--episode_length', '200',
            '--n_rollout_threads', '8',
            '--ppo_epoch', '10',
            '--num_mini_batch', '1',
            # Phase credit settings
            '--use_phase_credit',
            '--credit_method', config['method'],
            '--phase_method', 'fixed',
            '--phase_fixed_num', '3',
            '--credit_decay_lambda', '0.5',
            '--noop_action', '0',
            '--log_phase_credit',
            # Fast testing settings
            '--save_interval', '400000',
            '--log_interval', '80000',
            '--eval_interval', '800000',
            '--n_eval_rollout_threads', '1',
            '--eval_episodes', '3',
            '--user_name', 'test_user',
            '--use_wandb', 'false'
        ] + config['extra_args']
        
        print(f"Running: python -m onpolicy.scripts.train.train_mpe ... --credit_method {config['method']}")
        
        try:
            import subprocess
            result = subprocess.run(cmd, cwd='/home/Karan/crazy_projects/on-policy',
                                  capture_output=False, text=True, timeout=1200)  # 20 min timeout
            
            if result.returncode == 0:
                print(f"✅ {config['method']} test completed successfully!")
            else:
                print(f"❌ {config['method']} test failed with return code: {result.returncode}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ {config['method']} test timed out after 20 minutes")
        except Exception as e:
            print(f"❌ {config['method']} test failed: {e}")

if __name__ == "__main__":
    print("QUICK PHASE CREDIT TESTING")
    print("Choose test type:")
    print("1. Single proxy test (~2500 episodes)")
    print("2. Proxy vs Deep comparison (2x ~2000 episodes)")
    print("3. Just run the simple proxy test")
    
    choice = input("Enter choice (1-3) or press Enter for option 3: ").strip()
    
    if choice == "1":
        run_quick_mpe_test()
    elif choice == "2":
        run_quick_proxy_vs_deep_comparison()
    else:
        print("Running simple proxy test (default)...")
        run_quick_mpe_test()
