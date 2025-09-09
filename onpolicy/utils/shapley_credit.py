import numpy as np
from typing import List, Tuple, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAggregator:
    def __init__(self, decay_lambda: float = 0.5):
        self.decay_lambda = decay_lambda

    def weights(self, P: int) -> np.ndarray:
        idx = np.arange(P)
        # more weight to later phases: p=0..P-1 -> exp(-lambda*(P-1-p))
        w = np.exp(-self.decay_lambda * (P - 1 - idx))
        s = w.sum() + 1e-8
        return w / s

    def aggregate(self, phase_credits: List[np.ndarray]) -> np.ndarray:
        if len(phase_credits) == 0:
            return np.array([])
        P = len(phase_credits)
        W = self.weights(P)
        # credits shape per phase: [n_rollout_threads, n_agents]
        agg = None
        for p, c in enumerate(phase_credits):
            if agg is None:
                agg = W[p] * c
            else:
                agg = agg + W[p] * c
        return agg


class ShapleyCalculator:
    def __init__(self, num_mc: int = 16, noop_action: int = 0, rng: Optional[np.random.RandomState] = None):
        self.num_mc = num_mc
        self.noop_action = noop_action
        self.rng = rng or np.random.RandomState(0)

    def _is_one_hot(self, a_last: np.ndarray) -> bool:
        # detect one-hot by values in {0,1} and row-sum approx 1
        if a_last.ndim == 0:
            return False
        if not np.all((a_last == 0) | (a_last == 1)):
            return False
        row_sum = a_last.sum(axis=-1)
        return np.allclose(row_sum, 1.0)

    def estimate_phase_credit(
        self,
        rewards_slice: np.ndarray,        # shape [L, n_env, n_agents] or [L, n_env, 1]
        actions_slice: np.ndarray,        # shape [L, n_env, n_agents, act_dim] (discrete index vector or one-hot)
        mask_valid: Optional[np.ndarray] = None,  # shape [L, n_env] mask of valid steps
    ) -> np.ndarray:
        L = rewards_slice.shape[0]
        n_env = rewards_slice.shape[1]
        if rewards_slice.shape[-1] > 1:
            R = rewards_slice.sum(axis=-1)  # [L, n_env]
        else:
            R = rewards_slice[..., 0]       # [L, n_env]
        if mask_valid is None:
            mask_valid = np.ones((L, n_env), dtype=bool)

        n_agents = actions_slice.shape[2]
        credit = np.zeros((n_env, n_agents), dtype=np.float32)
        valid_counts = mask_valid.sum(axis=0).clip(min=1)

        for a in range(n_agents):
            a_last = actions_slice[..., a, :]
            if a_last.shape[-1] == 1:
                act_idx = a_last[..., 0].astype(int)
                credit[:, a] = (act_idx != self.noop_action).astype(np.float32).sum(axis=0) / valid_counts
            else:
                if self._is_one_hot(a_last):
                    chosen = np.argmax(a_last, axis=-1)
                    credit[:, a] = (chosen != self.noop_action).astype(np.float32).sum(axis=0) / valid_counts
                else:
                    non_noop = (a_last != self.noop_action).any(axis=-1)
                    credit[:, a] = non_noop.astype(np.float32).sum(axis=0) / valid_counts
        s = credit.sum(axis=1, keepdims=True) + 1e-8
        credit = credit / s
        return credit


class DeepShapleyNet(nn.Module):
    """Neural network for learning Shapley-like value functions"""
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 64):
        super().__init__()
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Feature extractor for observations
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim * num_agents, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Coalition encoder - takes subset indicators
        self.coalition_encoder = nn.Sequential(
            nn.Linear(num_agents, hidden_dim // 2),
            nn.ReLU()
        )
        
        # Value function head
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2 + hidden_dim // 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Individual credit heads
        self.credit_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim + hidden_dim // 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, 1)
            ) for _ in range(num_agents)
        ])

    def forward(self, obs, actions, coalition_mask):
        """
        obs: [batch, obs_dim] - joint observation
        actions: [batch, num_agents, action_dim] - agent actions
        coalition_mask: [batch, num_agents] - binary mask indicating which agents are active
        """
        batch_size = obs.shape[0]
        
        # Encode observation
        obs_features = self.obs_encoder(obs)  # [batch, hidden_dim]
        
        # Encode coalition
        coalition_features = self.coalition_encoder(coalition_mask.float())  # [batch, hidden_dim//2]
        
        # Encode joint actions
        joint_actions = actions.view(batch_size, -1)  # [batch, num_agents * action_dim]
        action_features = self.action_encoder(joint_actions)  # [batch, hidden_dim//2]
        
        # Combine features for value prediction
        value_input = torch.cat([obs_features, action_features, coalition_features], dim=-1)
        coalition_value = self.value_head(value_input)  # [batch, 1]
        
        # Compute individual credits
        credit_input = torch.cat([obs_features, action_features], dim=-1)
        credits = []
        for i in range(self.num_agents):
            agent_credit = self.credit_heads[i](credit_input)  # [batch, 1]
            credits.append(agent_credit)
        credits = torch.cat(credits, dim=-1)  # [batch, num_agents]
        
        return coalition_value, credits


class DeepShapleyCalculator:
    """Deep learning-based Shapley value approximator"""
    
    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, 
                 device: torch.device, learning_rate: float = 1e-3, 
                 num_coalitions: int = 8, buffer_size: int = 1000):
        self.num_agents = num_agents
        self.device = device
        self.num_coalitions = num_coalitions
        self.buffer_size = buffer_size
        
        # Initialize network
        self.network = DeepShapleyNet(obs_dim, action_dim, num_agents).to(device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Experience buffer for training
        self.obs_buffer = []
        self.action_buffer = []
        self.reward_buffer = []
        self.coalition_buffer = []
        
        # For generating random coalitions
        self.rng = np.random.RandomState(0)

    def _generate_coalitions(self, batch_size: int) -> np.ndarray:
        """Generate random coalitions for training"""
        coalitions = []
        for _ in range(batch_size):
            # Sample multiple coalitions per batch item
            batch_coalitions = []
            for _ in range(self.num_coalitions):
                # Random subset of agents (including empty and full coalitions)
                coalition_size = self.rng.randint(0, self.num_agents + 1)
                if coalition_size == 0:
                    coalition = np.zeros(self.num_agents, dtype=bool)
                elif coalition_size == self.num_agents:
                    coalition = np.ones(self.num_agents, dtype=bool)
                else:
                    coalition = np.zeros(self.num_agents, dtype=bool)
                    selected = self.rng.choice(self.num_agents, coalition_size, replace=False)
                    coalition[selected] = True
                batch_coalitions.append(coalition)
            coalitions.append(batch_coalitions)
        return np.array(coalitions)  # [batch, num_coalitions, num_agents]

    def add_experience(self, obs: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """Add experience to buffer for training"""
        self.obs_buffer.append(obs.copy())
        self.action_buffer.append(actions.copy())
        self.reward_buffer.append(rewards.copy())
        
        # Keep buffer size manageable
        if len(self.obs_buffer) > self.buffer_size:
            self.obs_buffer.pop(0)
            self.action_buffer.pop(0)
            self.reward_buffer.pop(0)

    def train_step(self, batch_size: int = 32):
        """Train the deep Shapley network"""
        if len(self.obs_buffer) < batch_size:
            return {}
        
        # Sample batch
        indices = self.rng.choice(len(self.obs_buffer), batch_size, replace=False)
        batch_obs = torch.FloatTensor([self.obs_buffer[i] for i in indices]).to(self.device)
        batch_actions = torch.FloatTensor([self.action_buffer[i] for i in indices]).to(self.device)
        batch_rewards = torch.FloatTensor([self.reward_buffer[i] for i in indices]).to(self.device)
        
        # Generate coalitions
        coalitions = self._generate_coalitions(batch_size)  # [batch, num_coalitions, num_agents]
        
        total_loss = 0.0
        value_loss = 0.0
        credit_loss = 0.0
        
        for coal_idx in range(self.num_coalitions):
            coalition_mask = torch.FloatTensor(coalitions[:, coal_idx]).to(self.device)
            
            # Forward pass
            pred_values, pred_credits = self.network(batch_obs, batch_actions, coalition_mask)
            
            # Compute target values (sum of rewards for active agents)
            active_rewards = batch_rewards * coalition_mask  # [batch, num_agents]
            target_values = active_rewards.sum(dim=-1, keepdim=True)  # [batch, 1]
            
            # Value function loss
            val_loss = F.mse_loss(pred_values, target_values)
            value_loss += val_loss
            
            # Credit consistency loss (credits should sum to value)
            masked_credits = pred_credits * coalition_mask  # [batch, num_agents]
            credit_sum = masked_credits.sum(dim=-1, keepdim=True)  # [batch, 1]
            cons_loss = F.mse_loss(credit_sum, target_values)
            credit_loss += cons_loss
            
            total_loss += val_loss + cons_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), 1.0)
        self.optimizer.step()
        
        return {
            'deep_shapley/total_loss': total_loss.item(),
            'deep_shapley/value_loss': value_loss.item(),
            'deep_shapley/credit_loss': credit_loss.item()
        }

    def estimate_phase_credit(self, 
                            obs_slice: np.ndarray,     # [L, n_env, obs_dim]
                            actions_slice: np.ndarray, # [L, n_env, n_agents, act_dim]
                            rewards_slice: np.ndarray, # [L, n_env, n_agents] or [L, n_env, 1]
                            mask_valid: Optional[np.ndarray] = None) -> np.ndarray:
        """Estimate phase credit using trained deep network"""
        
        L, n_env, n_agents = actions_slice.shape[:3]
        
        # Process rewards
        if rewards_slice.shape[-1] > 1:
            R = rewards_slice  # [L, n_env, n_agents]
        else:
            R = np.repeat(rewards_slice, n_agents, axis=-1)  # [L, n_env, n_agents]
        
        if mask_valid is None:
            mask_valid = np.ones((L, n_env), dtype=bool)
        
        # Add experiences to buffer for continued learning
        for t in range(L):
            for env_idx in range(n_env):
                if mask_valid[t, env_idx]:
                    self.add_experience(
                        obs_slice[t, env_idx],
                        actions_slice[t, env_idx],
                        R[t, env_idx]
                    )
        
        # Train network periodically
        if len(self.obs_buffer) >= 32:
            train_info = self.train_step()
        else:
            train_info = {}
        
        # Estimate credits using current network
        self.network.eval()
        with torch.no_grad():
            credits = np.zeros((n_env, n_agents), dtype=np.float32)
            
            for env_idx in range(n_env):
                # Use full coalition for final credit estimation
                full_coalition = np.ones(n_agents, dtype=bool)
                
                # Average over valid timesteps
                valid_steps = []
                for t in range(L):
                    if mask_valid[t, env_idx]:
                        obs_tensor = torch.FloatTensor(obs_slice[t, env_idx]).unsqueeze(0).to(self.device)
                        actions_tensor = torch.FloatTensor(actions_slice[t, env_idx]).unsqueeze(0).to(self.device)
                        coalition_tensor = torch.FloatTensor(full_coalition).unsqueeze(0).to(self.device)
                        
                        _, step_credits = self.network(obs_tensor, actions_tensor, coalition_tensor)
                        valid_steps.append(step_credits.cpu().numpy()[0])
                
                if valid_steps:
                    credits[env_idx] = np.mean(valid_steps, axis=0)
        
        # Normalize credits
        credit_sums = credits.sum(axis=1, keepdims=True) + 1e-8
        credits = credits / credit_sums
        
        self.network.train()
        return credits, train_info
