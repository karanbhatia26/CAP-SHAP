from collections import defaultdict, deque
from itertools import chain
import os
import time

import imageio
import numpy as np
import torch
try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner


def _t2n(x):
    return x.detach().cpu().numpy()

class FootballRunner(Runner):
    def __init__(self, config):
        super(FootballRunner, self).__init__(config)
        self.env_infos = defaultdict(list)
        
        # Phase credit initialization
        if self.all_args.use_phase_credit:
            from onpolicy.utils.phase_segmentation import PhaseSegmenter
            from onpolicy.utils.shapley_credit import ShapleyCalculator, DeepShapleyCalculator, TemporalAggregator
            
            self.phase_segmenter = PhaseSegmenter(
                method=self.all_args.phase_method,
                threshold=self.all_args.phase_threshold,
                min_length=self.all_args.phase_min_len,
                fixed_num_phases=self.all_args.phase_fixed_num,
                smoothing_window=self.all_args.phase_smooth_window
            )
            
            if self.all_args.credit_method == 'deep':
                # Get environment dimensions for deep Shapley
                obs_dim = self.envs.observation_space[0].shape[0] if hasattr(self.envs.observation_space[0], 'shape') else self.envs.observation_space[0].n
                action_dim = self.envs.action_space[0].n if hasattr(self.envs.action_space[0], 'n') else self.envs.action_space[0].shape[0]
                
                self.shapley_calculator = DeepShapleyCalculator(
                    obs_dim=obs_dim,
                    action_dim=action_dim,
                    num_agents=self.num_agents,
                    device=self.device,
                    learning_rate=self.all_args.deep_shapley_lr,
                    num_coalitions=self.all_args.deep_shapley_coalitions,
                    buffer_size=self.all_args.deep_shapley_buffer
                )
            else:
                self.shapley_calculator = ShapleyCalculator(
                    num_mc=16,
                    noop_action=self.all_args.noop_action,
                    rng=np.random.RandomState(42)
                )
            
            self.temporal_aggregator = TemporalAggregator(decay_lambda=self.all_args.credit_decay_lambda)
            
            # Episode tracking for phase credit
            self.ep_rewards = []
            self.ep_actions_idx = []
            self.ep_obs = []  # For deep Shapley
       
    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # Reset episode tracking for phase credit
            if self.all_args.use_phase_credit:
                self.ep_rewards = []
                self.ep_actions_idx = []
                self.ep_obs = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                    
                # Observe reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)

                # Track episode data for phase credit
                if self.all_args.use_phase_credit:
                    self.ep_rewards.append(rewards.copy())
                    self.ep_actions_idx.append(actions_env.copy())
                    self.ep_obs.append(self.envs.buf_obs.copy())  # Store observations for deep Shapley

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # Phase credit computation and buffer update
            if self.all_args.use_phase_credit:
                try:
                    credit_weights = self._compute_credit_weights(
                        np.array(self.ep_rewards), 
                        np.array(self.ep_actions_idx),
                        np.array(self.ep_obs) if self.all_args.credit_method == 'deep' else None
                    )
                    
                    # Update each agent's buffer with credit weights
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].update_credit(credit_weights)
                        
                    # Logging
                    if self.all_args.log_phase_credit:
                        self._log_phase_credit_stats(credit_weights, episode)
                        
                except Exception as e:
                    print(f"[WARNING] Phase credit computation failed at episode {episode}: {e}")
                    print(f"ep_rewards shape: {np.array(self.ep_rewards).shape if self.ep_rewards else 'empty'}")
                    print(f"ep_actions shape: {np.array(self.ep_actions_idx).shape if self.ep_actions_idx else 'empty'}")
                    import traceback
                    traceback.print_exc()

            # compute return and update network
            self.compute()
            train_infos = self.train()
            
            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                        .format(self.all_args.scenario_name,
                                self.algorithm_name,
                                self.experiment_name,
                                episode,
                                episodes,
                                total_num_steps,
                                self.num_env_steps,
                                int(total_num_steps / (end - start))))

                if self.env_name == "Football":
                    env_infos = self.process_infos(infos)
                    avg_ep_rew = np.mean(self.buffer[0].rewards) * self.episode_length
                    train_infos["average_episode_rewards"] = avg_ep_rew
                    print("average episode rewards is {}".format(avg_ep_rew))
                    
                self.log_train(train_infos, total_num_steps)
                self.log_env(env_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def _compute_credit_weights(self, ep_rewards, ep_actions_idx, ep_obs=None):
        """Compute credit weights using phase segmentation and Shapley estimation."""
        # ep_rewards: [T, n_env, n_agents] or [T, n_env, 1]
        # ep_actions_idx: [T, n_env, n_agents]
        # ep_obs: [T, n_env, obs_dim] (for deep Shapley)
        
        T, n_env = ep_rewards.shape[:2]
        n_agents = self.num_agents
        
        # Aggregate rewards across episodes/environments for phase segmentation
        if ep_rewards.shape[-1] == 1:
            # Shared reward case
            reward_signal = ep_rewards[:, :, 0].mean(axis=1)  # [T]
        else:
            # Individual rewards
            reward_signal = ep_rewards.mean(axis=(1, 2))  # [T]
        
        # Phase segmentation
        phases = self.phase_segmenter.segment(reward_signal)
        
        # Compute credit per phase
        phase_credits = []
        deep_shapley_infos = []
        
        for start_idx, end_idx in phases:
            if self.all_args.credit_method == 'deep' and ep_obs is not None:
                # Deep Shapley credit
                phase_credit, train_info = self.shapley_calculator.estimate_phase_credit(
                    obs_slice=ep_obs[start_idx:end_idx],
                    actions_slice=ep_actions_idx[start_idx:end_idx],
                    rewards_slice=ep_rewards[start_idx:end_idx]
                )
                deep_shapley_infos.append(train_info)
            else:
                # Proxy Shapley credit
                phase_credit = self.shapley_calculator.estimate_phase_credit(
                    rewards_slice=ep_rewards[start_idx:end_idx],
                    actions_slice=ep_actions_idx[start_idx:end_idx]
                )
            
            phase_credits.append(phase_credit)
        
        # Temporal aggregation
        aggregated_credit = self.temporal_aggregator.aggregate(phase_credits)  # [n_env, n_agents]
        
        # Expand to per-step weights
        credit_weights = np.repeat(aggregated_credit[np.newaxis, :, :], T, axis=0)  # [T, n_env, n_agents]
        
        # Log deep Shapley training info if available
        if deep_shapley_infos and self.all_args.log_phase_credit:
            for i, info in enumerate(deep_shapley_infos):
                for key, value in info.items():
                    if self.writter is not None:
                        self.writter.add_scalar(f"phase_{i}/{key}", value, self.total_num_steps)
        
        return credit_weights

    def _log_phase_credit_stats(self, credit_weights, episode):
        """Log phase credit statistics for debugging/analysis."""
        # credit_weights: [T, n_env, n_agents]
        
        if self.writter is None:
            return
            
        # Basic stats
        mean_weight = credit_weights.mean()
        std_weight = credit_weights.std()
        min_weight = credit_weights.min()
        max_weight = credit_weights.max()
        
        self.writter.add_scalar("credit/weight_mean", mean_weight, episode)
        self.writter.add_scalar("credit/weight_std", std_weight, episode)
        self.writter.add_scalar("credit/weight_min", min_weight, episode)
        self.writter.add_scalar("credit/weight_max", max_weight, episode)
        
        # Per-agent average credit
        for agent_id in range(self.num_agents):
            agent_avg_credit = credit_weights[:, :, agent_id].mean()
            self.writter.add_scalar(f"credit/agent_credit/agent_{agent_id}", agent_avg_credit, episode)
        
        # Optional histogram
        if self.all_args.log_phase_hist:
            self.writter.add_histogram("credit/weight_histogram", credit_weights.flatten(), episode)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                          self.buffer[agent_id].obs[step],
                                                          self.buffer[agent_id].rnn_states[step],
                                                          self.buffer[agent_id].rnn_states_critic[step],
                                                          self.buffer[agent_id].masks[step])
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                for i in range(self.envs.action_space[agent_id].shape):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i] + 1)[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
            else:
                raise NotImplementedError
            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for agent_id in range(self.num_agents):
                if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    one_hot_action_env.append(temp_actions_env[agent_id][i])
                elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    one_hot_action_env.append(temp_actions_env[agent_id][i])
                else:
                    raise NotImplementedError
            actions_env.append(one_hot_action_env)
        actions_env = np.array(actions_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = np.all(dones, axis=1)

        rnn_states[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[3:]), dtype=np.float32)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(share_obs,
                                       np.array(list(obs[:, agent_id])),
                                       rnn_states[:, agent_id],
                                       rnn_states_critic[:, agent_id],
                                       actions[:, agent_id],
                                       action_log_probs[:, agent_id],
                                       values[:, agent_id],
                                       rewards[:, agent_id],
                                       masks[:, agent_id])

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            train_infos[agent_id]["average_step_rewards"] = np.mean(self.buffer[agent_id].rewards)
        super().log_train(train_infos, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                if self.use_wandb and _HAS_WANDB:
                    wandb.log({k: np.mean(v)}, step=total_num_steps)
                else:
                    self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros((self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(np.array(list(eval_obs[:, agent_id])),
                                                                              eval_rnn_states[:, agent_id],
                                                                              eval_masks[:, agent_id],
                                                                              deterministic=True)
                eval_action = _t2n(eval_action)
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i] + 1)[eval_action[:, i]]
                        if i == 0:
                            eval_action_env = eval_uc_action_env
                        else:
                            eval_action_env = np.concatenate((eval_action_env, eval_uc_action_env), axis=1)
                elif self.eval_envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                    eval_action_env = np.squeeze(np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1)
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)
                
            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for agent_id in range(self.num_agents):
                    eval_one_hot_action_env.append(eval_temp_actions_env[agent_id][i])
                eval_actions_env.append(eval_one_hot_action_env)
            eval_actions_env = np.array(eval_actions_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.num_agents, 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_env_infos = self.process_infos(eval_infos)
        
        eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards, axis=0))
        print("eval average episode rewards of agent: " + str(eval_average_episode_rewards))
        self.log_env(eval_env_infos, total_num_steps)

        if self.use_wandb and _HAS_WANDB:
            wandb.log({"eval_average_episode_rewards": eval_average_episode_rewards}, step=total_num_steps)
        else:
            self.writter.add_scalars("eval_average_episode_rewards", {"eval_average_episode_rewards": eval_average_episode_rewards}, total_num_steps)

    def process_infos(self, infos):
        env_infos = {}
        for agent_id in range(self.num_agents):
            idv_rews = []
            dist_goals, dist_ball, dist_goal_line, ball_owned, left_team = [], [], [], [], []
            for info in infos:
                if 'individual_reward' in info[agent_id].keys():
                    idv_rews.append(info[agent_id]['individual_reward'])
                if 'ball_dist' in info[agent_id].keys():
                    dist_ball.append(info[agent_id]['ball_dist'])
                if 'goal_dist' in info[agent_id].keys():
                    dist_goals.append(info[agent_id]['goal_dist'])
                if 'goal_line_dist' in info[agent_id].keys():
                    dist_goal_line.append(info[agent_id]['goal_line_dist'])
                if 'ball_owned' in info[agent_id].keys():
                    ball_owned.append(info[agent_id]['ball_owned'])
                if 'left_team' in info[agent_id].keys():
                    left_team.append(info[agent_id]['left_team'])
            agent_rew = f'agent{agent_id}/individual_rewards'
            agent_ball_dist = f'agent{agent_id}/ball_dist'
            agent_goal_dist = f'agent{agent_id}/goal_dist'
            agent_goal_line_dist = f'agent{agent_id}/goal_line_dist'
            agent_ball_owned = f'agent{agent_id}/ball_owned'
            agent_left_team = f'agent{agent_id}/left_team'

            env_infos[agent_rew] = idv_rews
            env_infos[agent_ball_dist] = dist_ball
            env_infos[agent_goal_dist] = dist_goals
            env_infos[agent_goal_line_dist] = dist_goal_line
            env_infos[agent_ball_owned] = ball_owned
            env_infos[agent_left_team] = left_team
        return env_infos
