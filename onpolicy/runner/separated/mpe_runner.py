import time
import os
import numpy as np
from itertools import chain
import torch

# Optional Weights & Biases
try:
    import wandb  # type: ignore
    HAS_WANDB = True
except Exception:
    HAS_WANDB = False

from onpolicy.utils.util import update_linear_schedule
from onpolicy.runner.separated.base_runner import Runner
import imageio

# Added: phase segmentation + credit
from onpolicy.utils.phase_segmentation import PhaseDetector
from onpolicy.utils.shapley_credit import ShapleyCalculator, TemporalAggregator

def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)
        # credit assignment components
        self.use_phase_credit = self.all_args.use_phase_credit
        if self.use_phase_credit:
            self.phase_detector = PhaseDetector(
                method=self.all_args.phase_method,
                threshold=self.all_args.phase_threshold,
                min_len=self.all_args.phase_min_len,
                fixed_num_phases=self.all_args.phase_fixed_num,
                smooth_window=self.all_args.phase_smooth_window,
            )
            self.shapley_calc = ShapleyCalculator(noop_action=getattr(self.all_args, 'noop_action', None))
            self.temporal_agg = TemporalAggregator(decay_lambda=self.all_args.credit_decay_lambda)

    def run(self):
        self.warmup()   

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)

            # rollout storage for per-episode logging and credit calc
            ep_rewards = []  # shape [T, N, A] from env
            # actions as indices for credit proxy
            ep_actions_idx = []  # shape [T, N, A, 1]

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env = self.collect(step)
                # store index actions per agent for credit proxy
                ep_actions_idx.append(actions.copy())
                
                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                ep_rewards.append(rewards.copy())

                data = obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic 
                
                # insert data into buffer
                self.insert(data)

            # compute return and update network
            # Before compute/train, optionally compute credit weights from the rollout
            if self.use_phase_credit:
                try:
                    credit_weights = self._compute_credit_weights(np.array(ep_rewards), np.array(ep_actions_idx))
                    # update each agent buffer with per-step scalar weights
                    for agent_id in range(self.num_agents):
                        self.buffer[agent_id].update_credit(credit_weights)
                    
                    # Add explicit logging every few episodes
                    if episode % 20 == 0:  # Log every 20 episodes
                        mean_credit = credit_weights.mean()
                        std_credit = credit_weights.std()
                        print(f"[PHASE_CREDIT] Episode {episode}: credit_weights shape={credit_weights.shape}, mean={mean_credit:.4f}, std={std_credit:.4f}")
                        print(f"[PHASE_CREDIT] Agent credits: {credit_weights.mean(axis=(0,1))}")
                        
                except Exception as e:
                    import traceback
                    print(f"[phase_credit] warning: failed to compute credit weights: {e}")
                    print(f"[phase_credit] ep_rewards.shape={np.array(ep_rewards).shape}, ep_actions_idx.shape={np.array(ep_actions_idx).shape}")
                    traceback.print_exc()

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

                if self.env_name == "MPE":
                    for agent_id in range(self.num_agents):
                        idv_rews = []
                        for info in infos:
                            for count, info in enumerate(infos):
                                if 'individual_reward' in infos[count][agent_id].keys():
                                    idv_rews.append(infos[count][agent_id].get('individual_reward', 0))
                        train_infos[agent_id].update({'individual_rewards': np.mean(idv_rews)})
                        train_infos[agent_id].update({"average_episode_rewards": np.mean(self.buffer[agent_id].rewards) * self.episode_length})
                # extra logging: average number of phases if enabled
                if self.use_phase_credit and hasattr(self, '_last_phase_counts'):
                    if getattr(self, 'use_wandb', False) and HAS_WANDB:
                        wandb.log({'phase/avg_phases': np.mean(self._last_phase_counts)}, step=total_num_steps)
                    else:
                        self.writter.add_scalars('phase/avg_phases', {'phase/avg_phases': np.mean(self._last_phase_counts)}, total_num_steps)
                # phase credit diagnostics logging
                if self.use_phase_credit and hasattr(self, '_last_credit_diag') and getattr(self.all_args, 'log_phase_credit', False):
                    stats = self._last_credit_diag
                    if getattr(self, 'use_wandb', False) and HAS_WANDB:
                        log_dict = {
                            'credit/weight_mean': stats['w_mean'],
                            'credit/weight_std': stats['w_std'],
                            'credit/weight_min': stats['w_min'],
                            'credit/weight_max': stats['w_max'],
                        }
                        for i, v in enumerate(stats['agent_credit']):
                            log_dict[f'credit/agent_{i}'] = float(v)
                        wandb.log(log_dict, step=total_num_steps)
                        if getattr(self.all_args, 'log_phase_hist', False):
                            wandb.log({
                                'credit/weights_hist': wandb.Histogram(np.asarray(self._last_credit_weights_for_logging)),
                                'phase/lengths_hist': wandb.Histogram(np.asarray(stats['phase_lengths']))
                            }, step=total_num_steps)
                    else:
                        self.writter.add_scalars('credit/weights_stats', {
                            'mean': stats['w_mean'],
                            'std': stats['w_std'],
                            'min': stats['w_min'],
                            'max': stats['w_max'],
                        }, total_num_steps)
                        ac = {f'agent_{i}': float(v) for i, v in enumerate(stats['agent_credit'])}
                        self.writter.add_scalars('credit/agent_credit', ac, total_num_steps)
                        if getattr(self.all_args, 'log_phase_hist', False):
                            self.writter.add_histogram('credit/weights', np.asarray(self._last_credit_weights_for_logging), total_num_steps)
                            if np.asarray(stats['phase_lengths']).size:
                                self.writter.add_histogram('phase/lengths', np.asarray(stats['phase_lengths']), total_num_steps)
                self.log_train(train_infos, total_num_steps)

            # eval
            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

    def warmup(self):
        # reset env
        obs = self.envs.reset()

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

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
                    uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
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
            rnn_states_critic.append( _t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

        return values, actions, action_log_probs, rnn_states, rnn_states_critic, actions_env

    def insert(self, data):
        obs, rewards, dones, infos, values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        rnn_states_critic[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

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

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                    for i in range(self.eval_envs.action_space[agent_id].shape):
                        eval_uc_action_env = np.eye(self.eval_envs.action_space[agent_id].high[i]+1)[eval_action[:, i]]
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
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(((eval_dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)
        
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            eval_average_episode_rewards = np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0))
            eval_train_infos.append({'eval_average_episode_rewards': eval_average_episode_rewards})
            print("eval average episode rewards of agent%i: " % agent_id + str(eval_average_episode_rewards))

        self.log_train(eval_train_infos, total_num_steps)  

    @torch.no_grad()
    def render(self):        
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render('rgb_array')[0][0]
                all_frames.append(image)

            rnn_states = np.zeros((self.n_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size), dtype=np.float32)
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            for step in range(self.episode_length):
                calc_start = time.time()
                
                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(np.array(list(obs[:, agent_id])),
                                                                        rnn_states[:, agent_id],
                                                                        masks[:, agent_id],
                                                                        deterministic=True)

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == 'MultiDiscrete':
                        for i in range(self.envs.action_space[agent_id].shape):
                            uc_action_env = np.eye(self.envs.action_space[agent_id].high[i]+1)[action[:, i]]
                            if i == 0:
                                action_env = uc_action_env
                            else:
                                action_env = np.concatenate((action_env, uc_action_env), axis=1)
                    elif self.envs.action_space[agent_id].__class__.__name__ == 'Discrete':
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)
                   
                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(((dones == True).sum(), self.recurrent_N, self.hidden_size), dtype=np.float32)
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render('rgb_array')[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))
        
        if self.all_args.save_gifs:
            imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)

    # ----- Phase segmentation + Shapley-style credit (proxy) -----
    def _compute_credit_weights(self, ep_rewards: np.ndarray, ep_actions_idx: np.ndarray) -> np.ndarray:
        """
        Inputs:
          ep_rewards: [T, N, A] or [T, N, A, 1] as returned by env.step (per-agent rewards in MPE). We'll sum over A.
          ep_actions_idx: [T, N, A, act_dim] index representation from collect().
        Returns:
          credit_weights: [T, N, 1] to be applied to each agent's advantages uniformly (shared scalar per env-step).
        """
        # Handle extra singleton reward dim
        if ep_rewards.ndim == 4 and ep_rewards.shape[-1] == 1:
            ep_rewards = ep_rewards[..., 0]
        T, N, A = ep_rewards.shape
        # derive global reward per step per env
        Rg = ep_rewards.sum(axis=2)  # [T, N]
        # phase segmentation per env
        phases_per_env = []
        for env_i in range(N):
            segs = self.phase_detector.segment_episode(Rg[:, env_i])
            phases_per_env.append(segs)
        self._last_phase_counts = np.array([len(segs) for segs in phases_per_env])

        # compute per-phase proxy shapley credits per env
        phase_credits_env = [[] for _ in range(N)]
        for env_i in range(N):
            segs = phases_per_env[env_i]
            for (ps, pe) in segs:
                rewards_slice = ep_rewards[ps:pe, env_i:env_i+1, :]  # [L, 1, A]
                actions_slice = ep_actions_idx[ps:pe, env_i:env_i+1, :, :]  # [L, 1, A, act_dim]
                credit_phase = self.shapley_calc.estimate_phase_credit(rewards_slice, actions_slice)  # [1, A]
                phase_credits_env[env_i].append(credit_phase)  # list of [1,A]

        # aggregate across phases with temporal weights
        # For each env: list of [1,A] -> aggregate -> [1,A]
        per_env_agent_credit = np.zeros((N, A), dtype=np.float32)
        for env_i in range(N):
            if len(phase_credits_env[env_i]) == 0:
                per_env_agent_credit[env_i] = 1.0 / A
            else:
                agg = self.temporal_agg.aggregate(phase_credits_env[env_i])  # [1, A]
                per_env_agent_credit[env_i] = agg[0]
        # map agent credits to a scalar per step per env. Simple choice: average over agents engaged at that step.
        credit_weights = np.zeros((T, N, 1), dtype=np.float32)
        # normalize so average weight per env over time is 1
        for env_i in range(N):
            w_env = per_env_agent_credit[env_i].mean()  # scalar
            credit_weights[:, env_i, 0] = w_env
        # rescale to mean 1
        mean_w = credit_weights.mean()
        if mean_w > 0:
            credit_weights /= mean_w

        # diagnostics for logging
        flat_w = credit_weights.reshape(-1)
        agent_credit = per_env_agent_credit.mean(axis=0)
        sum_ac = agent_credit.sum()
        if sum_ac > 0:
            agent_credit = agent_credit / sum_ac
        phase_lengths = np.array([pe - ps for segs in phases_per_env for (ps, pe) in segs], dtype=np.int32)
        self._last_credit_diag = {
            'w_mean': float(flat_w.mean()),
            'w_std': float(flat_w.std()),
            'w_min': float(flat_w.min()) if flat_w.size else 0.0,
            'w_max': float(flat_w.max()) if flat_w.size else 0.0,
            'agent_credit': agent_credit,
            'phase_lengths': phase_lengths,
        }
        # small sample for histogram logging
        try:
            max_elems = getattr(self.all_args, 'log_hist_max_elems', 5000)
        except Exception:
            max_elems = 5000
        if flat_w.size > max_elems:
            idx = np.linspace(0, flat_w.size - 1, max_elems, dtype=np.int64)
            self._last_credit_weights_for_logging = flat_w[idx]
        else:
            self._last_credit_weights_for_logging = flat_w

        return credit_weights
