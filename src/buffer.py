"""
Rollout Buffer for MAPPO
==========================
Stores transitions from multi-agent rollouts and computes
Generalised Advantage Estimation (GAE) for PPO updates.
"""

import numpy as np
import torch
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAPPO_GAMMA, MAPPO_GAE_LAMBDA


class RolloutBuffer:
    """
    Multi-agent rollout buffer.

    Stores per-agent transitions and computes GAE advantages using
    the centralised value function.  Data is later consumed by
    MAPPOAgent.update() via get_batches().
    """

    def __init__(self, rollout_length, n_agents, obs_dim, state_dim,
                 action_dim, gamma=MAPPO_GAMMA, gae_lambda=MAPPO_GAE_LAMBDA):
        self.rollout_length = rollout_length
        self.n_agents       = n_agents
        self.obs_dim        = obs_dim
        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.gamma          = gamma
        self.gae_lambda     = gae_lambda

        self._allocate()

    # ----------------------------------------------------------
    # Allocation / reset
    # ----------------------------------------------------------
    def _allocate(self):
        T, N = self.rollout_length, self.n_agents

        self.obs       = np.zeros((T, N, self.obs_dim),    dtype=np.float32)
        self.states    = np.zeros((T, self.state_dim),     dtype=np.float32)
        self.actions   = np.zeros((T, N, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((T, N),                  dtype=np.float32)
        self.rewards   = np.zeros((T, N),                  dtype=np.float32)
        self.dones     = np.zeros((T,),                    dtype=np.float32)
        self.values    = np.zeros((T, N),                  dtype=np.float32)

        # Computed by compute_gae()
        self.advantages = np.zeros((T, N), dtype=np.float32)
        self.returns    = np.zeros((T, N), dtype=np.float32)

        self.step = 0

    def reset(self):
        self._allocate()

    # ----------------------------------------------------------
    # Data insertion
    # ----------------------------------------------------------
    def insert(self, obs, state, actions, log_probs, rewards, done, values):
        """
        Insert one timestep.

        Args
        ----
        obs       : (n_agents, obs_dim)
        state     : (state_dim,)
        actions   : (n_agents, action_dim)
        log_probs : (n_agents,)
        rewards   : (n_agents,)
        done      : float  (1.0 = episode ended)
        values    : (n_agents,)
        """
        t = self.step
        self.obs[t]       = obs
        self.states[t]    = state
        self.actions[t]   = actions
        self.log_probs[t] = log_probs
        self.rewards[t]   = rewards
        self.dones[t]     = done
        self.values[t]    = values
        self.step        += 1

    # ----------------------------------------------------------
    # GAE computation
    # ----------------------------------------------------------
    def compute_gae(self, last_values, last_done):
        """
        Compute GAE advantages and discounted returns.

        Args
        ----
        last_values : (n_agents,) — V(s_{T+1})
        last_done   : float
        """
        gae = np.zeros(self.n_agents, dtype=np.float32)

        for t in reversed(range(self.rollout_length)):
            if t == self.rollout_length - 1:
                next_values = last_values
            else:
                next_values = self.values[t + 1]

            # Use dones[t] so the terminal bootstrap mask is applied correctly.
            mask  = 1.0 - self.dones[t]
            delta = (self.rewards[t]
                     + self.gamma * next_values * mask
                     - self.values[t])
            gae   = delta + self.gamma * self.gae_lambda * mask * gae

            self.advantages[t] = gae
            self.returns[t]    = self.advantages[t] + self.values[t]

    # ----------------------------------------------------------
    # Mini-batch generator
    # ----------------------------------------------------------
    def get_batches(self, num_mini_batches):
        """
        Yield mini-batches for PPO training.

        Flattens the agent dimension into the batch dimension so a single
        actor update covers all agents (parameter sharing). Uses pinned-memory
        tensors to overlap CPU→GPU transfer with computation.

        Advantages are not normalised here — normalisation is done once per
        update call in MAPPOAgent.update().

        Yields dicts of CPU-pinned tensors with shape (mini_batch_size, ...).
        """
        T, N  = self.rollout_length, self.n_agents
        total = T * N
        mini_batch_size = total // num_mini_batches

        # Flatten agent dimension into batch dimension
        obs_flat        = self.obs.reshape(total, -1)
        states_flat     = np.repeat(self.states, N, axis=0)
        actions_flat    = self.actions.reshape(total, -1)
        log_probs_flat  = self.log_probs.reshape(total)
        advantages_flat = self.advantages.reshape(total)
        returns_flat    = self.returns.reshape(total)
        values_flat     = self.values.reshape(total)

        # Pre-convert to pinned CPU tensors for efficient GPU transfer
        def _pin(arr):
            t = torch.from_numpy(arr)
            try:
                return t.pin_memory()
            except Exception:
                return t

        obs_t        = _pin(obs_flat)
        states_t     = _pin(states_flat)
        actions_t    = _pin(actions_flat)
        log_probs_t  = _pin(log_probs_flat)
        advantages_t = _pin(advantages_flat)
        returns_t    = _pin(returns_flat)
        values_t     = _pin(values_flat)

        indices = np.random.permutation(total)

        for start in range(0, total, mini_batch_size):
            end    = min(start + mini_batch_size, total)
            mb_idx = indices[start:end]
            agent_ids = torch.from_numpy((mb_idx % N).astype(np.int64))

            yield {
                'obs':           obs_t[mb_idx],
                'states':        states_t[mb_idx],
                'actions':       actions_t[mb_idx],
                'old_log_probs': log_probs_t[mb_idx],
                'advantages':    advantages_t[mb_idx],
                'returns':       returns_t[mb_idx],
                'old_values':    values_t[mb_idx],
                'agent_ids':     agent_ids,
            }
