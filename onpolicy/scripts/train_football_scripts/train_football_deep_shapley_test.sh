#!/bin/sh
# exp param
env="Football"
scenario="academy_3_vs_1_with_keeper"
algo="mappo" # "mappo" "ippo" "rmappo"
exp="deep_shapley_test"

# football param
num_agents=3

# train param - REDUCED FOR FAST TESTING
num_env_steps=3000000  # Reduced from 25M to 3M for faster testing (~2000-3000 episodes)
episode_length=200
n_rollout_threads=50
ppo_epoch=15
num_mini_batch=2

# Phase credit params - DEEP SHAPLEY
use_phase_credit=true
credit_method="deep"
phase_method="fixed"
phase_fixed_num=3
credit_decay_lambda=0.5
noop_action=0

# Deep Shapley specific params
deep_shapley_lr=1e-3
deep_shapley_hidden=64
deep_shapley_buffer=1000
deep_shapley_coalitions=8

# Logging
log_phase_credit=true
log_phase_hist=false
save_interval=50000   # More frequent saves for shorter runs
log_interval=25000    # More frequent logging
eval_interval=100000  # Reduced eval interval

echo "Running Football deep Shapley test with neural network credit assignment"
echo "n_rollout_threads: ${n_rollout_threads} \t ppo_epoch: ${ppo_epoch} \t num_mini_batch: ${num_mini_batch}"

CUDA_VISIBLE_DEVICES=0 python ../train/train_football.py \
--env_name ${env} --scenario_name ${scenario} --algorithm_name ${algo} --experiment_name ${exp} --seed 1 \
--num_agents ${num_agents} --num_env_steps ${num_env_steps} --episode_length ${episode_length} \
--representation "simple115v2" --rewards "scoring,checkpoints" \
--n_rollout_threads ${n_rollout_threads} --ppo_epoch ${ppo_epoch} --num_mini_batch ${num_mini_batch} \
--use_phase_credit --credit_method ${credit_method} --phase_method ${phase_method} \
--phase_fixed_num ${phase_fixed_num} --credit_decay_lambda ${credit_decay_lambda} --noop_action ${noop_action} \
--deep_shapley_lr ${deep_shapley_lr} --deep_shapley_hidden ${deep_shapley_hidden} \
--deep_shapley_buffer ${deep_shapley_buffer} --deep_shapley_coalitions ${deep_shapley_coalitions} \
--log_phase_credit --log_phase_hist=${log_phase_hist} \
--save_interval ${save_interval} --log_interval ${log_interval} --use_eval --eval_interval ${eval_interval} \
--n_eval_rollout_threads 20 --eval_episodes 10 \
--user_name "test_user" --use_wandb false
