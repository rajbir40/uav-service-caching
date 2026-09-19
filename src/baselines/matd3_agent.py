"""
MATD3 Baseline  (Table I — Row 4)
=====================================
Double-Q Deterministic Multi-Agent DRL (multi-agent TD3 / Fujimoto et al., 2018).

Key characteristics vs MADDPG:
  • Twin centralised Q-critics per agent  → reduces overestimation bias
  • Delayed policy updates               → actor updated every d critic steps
  • Target policy smoothing             → noise added to target actions
  • No diffusion, no NOMA optimisation in the policy

Interface (identical to MAPPOAgent):
    get_actions(obs_all, state, num_active) → actions, log_probs(=0), values(=0)
    get_values(state)                       → zeros
    update(buffer)                          → metrics dict
    save(path) / load(path)
"""

import os
import sys
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config import (
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_MAX_GRAD_NORM,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
)

# ── Hyper-parameters ────────────────────────────────────────────
MATD3_GAMMA           = 0.99
MATD3_TAU             = 0.005    # Polyak averaging
MATD3_NOISE_STD       = 0.1     # Exploration noise
MATD3_POLICY_NOISE    = 0.2     # Target policy smoothing noise
MATD3_NOISE_CLIP      = 0.5     # Clip target policy noise
MATD3_POLICY_DELAY    = 2       # Actor update every d critic steps
MATD3_BATCH_SIZE      = 256
MATD3_REPLAY_CAP      = 200_000


# ================================================================
# DETERMINISTIC ACTOR  μ_θ(o) → a
# ================================================================
class _DeterministicActor(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=MAPPO_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),    nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, action_dim), nn.Tanh(),
        )

    def forward(self, obs):
        return self.net(obs)


# ================================================================
# TWIN Q-CRITICS  Q1, Q2(s, a_all) → scalar each
# ================================================================
class _TwinQCritic(nn.Module):
    """Two independent Q-networks sharing input — TD3's core component."""

    def __init__(self, state_dim, action_dim, n_agents,
                 hidden_dim=MAPPO_CRITIC_HIDDEN_DIM):
        super().__init__()
        in_dim = state_dim + action_dim * n_agents

        self.q1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),     nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )
        self.q2 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),     nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, actions_all):
        x = torch.cat([state, actions_all], dim=-1)
        return self.q1(x).squeeze(-1), self.q2(x).squeeze(-1)

    def q1_only(self, state, actions_all):
        x = torch.cat([state, actions_all], dim=-1)
        return self.q1(x).squeeze(-1)


# ================================================================
# SIMPLE REPLAY BUFFER (off-policy)
# ================================================================
class _ReplayBuffer:
    def __init__(self, capacity, n_agents, obs_dim, state_dim, action_dim):
        self.cap  = capacity
        self.N    = n_agents
        self.ptr  = 0
        self.size = 0

        self.obs     = np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.state   = np.zeros((capacity, state_dim),            dtype=np.float32)
        self.actions = np.zeros((capacity, n_agents, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, n_agents),             dtype=np.float32)
        self.next_obs= np.zeros((capacity, n_agents, obs_dim),    dtype=np.float32)
        self.next_st = np.zeros((capacity, state_dim),            dtype=np.float32)
        self.dones   = np.zeros((capacity,),                      dtype=np.float32)

    def push(self, obs, state, actions, rewards, next_obs, next_state, done):
        i = self.ptr % self.cap
        self.obs[i]      = obs
        self.state[i]    = state
        self.actions[i]  = actions
        self.rewards[i]  = rewards
        self.next_obs[i] = next_obs
        self.next_st[i]  = next_state
        self.dones[i]    = float(done)
        self.ptr  += 1
        self.size  = min(self.size + 1, self.cap)

    def sample(self, batch_size, device):
        idx = np.random.randint(0, self.size, size=batch_size)
        def _t(arr): return torch.from_numpy(arr[idx]).to(device)
        return (_t(self.obs), _t(self.state), _t(self.actions),
                _t(self.rewards), _t(self.next_obs), _t(self.next_st),
                _t(self.dones))

    def __len__(self): return self.size


# ================================================================
# MATD3 AGENT
# ================================================================
class MATD3Agent:
    """
    Multi-Agent Twin-Delayed DDPG (Row 4).

    One deterministic actor + one twin Q-critic per agent.
    Critic trained every step; actor + targets updated every d steps.
    """

    METHOD = "MATD3"

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device="cpu", share_actor=False):
        self.obs_dim    = obs_dim
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.device     = device

        # ── Actors ───────────────────────────────────────────────
        self.actors        = [_DeterministicActor(obs_dim, action_dim).to(device)
                              for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a) for a in self.actors]
        for ta in self.target_actors:
            ta.requires_grad_(False)

        # ── Twin Critics ─────────────────────────────────────────
        self.critics        = [_TwinQCritic(state_dim, action_dim,
                                             n_agents).to(device)
                               for _ in range(n_agents)]
        self.target_critics = [copy.deepcopy(c) for c in self.critics]
        for tc in self.target_critics:
            tc.requires_grad_(False)

        # ── Optimisers ───────────────────────────────────────────
        self.actor_opts  = [optim.Adam(a.parameters(), lr=MAPPO_LR_ACTOR)
                            for a in self.actors]
        self.critic_opts = [optim.Adam(c.parameters(), lr=MAPPO_LR_CRITIC)
                            for c in self.critics]

        # ── Replay buffer ─────────────────────────────────────────
        self.replay = _ReplayBuffer(MATD3_REPLAY_CAP, n_agents,
                                    obs_dim, state_dim, action_dim)

        self.total_updates = 0
        self._critic_steps = 0   # tracks when to update actors

    # ----------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------
    @torch.inference_mode()
    def get_actions(self, obs_all, state, num_active=None):
        n_active  = num_active or self.n_agents
        actions   = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values    = np.zeros(self.n_agents, dtype=np.float32)

        for i in range(n_active):
            obs_t = torch.from_numpy(
                np.ascontiguousarray(obs_all[i], dtype=np.float32)
            ).unsqueeze(0).to(self.device)
            act   = self.actors[i % len(self.actors)](obs_t).cpu().numpy().flatten()
            noise = np.random.normal(0, MATD3_NOISE_STD, act.shape)
            actions[i] = np.clip(act + noise, -1.0, 1.0)

        return actions, log_probs, values

    @torch.inference_mode()
    def get_values(self, state):
        return np.zeros(self.n_agents, dtype=np.float32)

    # ----------------------------------------------------------
    def push_transition(self, obs, state, actions, rewards,
                        next_obs, next_state, done):
        self.replay.push(obs, state, actions, rewards,
                         next_obs, next_state, done)

    # ----------------------------------------------------------
    # MATD3 Update
    # ----------------------------------------------------------
    def update(self, buffer=None) -> dict:
        if len(self.replay) < MATD3_BATCH_SIZE:
            return dict(actor_loss=0.0, critic_loss=0.0,
                        entropy=0.0, approx_kl=0.0)

        self.total_updates  += 1
        self._critic_steps  += 1

        (obs, state, actions, rewards, next_obs,
         next_state, dones) = self.replay.sample(MATD3_BATCH_SIZE, self.device)

        B, N = obs.shape[0], self.n_agents

        # ── Target actions with smoothing noise ───────────────────
        with torch.no_grad():
            next_acts_list = []
            for i in range(N):
                ta  = self.target_actors[i % len(self.target_actors)]
                na  = ta(next_obs[:, i, :])
                eps = torch.randn_like(na).clamp(-MATD3_NOISE_CLIP,
                                                  MATD3_NOISE_CLIP) * MATD3_POLICY_NOISE
                next_acts_list.append((na + eps).clamp(-1.0, 1.0))
            next_acts_all = torch.cat(next_acts_list, dim=-1)

        critic_loss_total = 0.0
        actor_loss_total  = 0.0

        for i in range(N):
            acts_flat = actions.reshape(B, N * self.action_dim)

            # ── Twin critic update ─────────────────────────────────
            q1, q2 = self.critics[i](state, acts_flat)

            with torch.no_grad():
                tq1, tq2 = self.target_critics[i](next_state, next_acts_all)
                q_target  = (rewards[:, i]
                             + MATD3_GAMMA * (1 - dones)
                             * torch.min(tq1, tq2))

            c_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)
            self.critic_opts[i].zero_grad()
            c_loss.backward()
            nn.utils.clip_grad_norm_(self.critics[i].parameters(),
                                     MAPPO_MAX_GRAD_NORM)
            self.critic_opts[i].step()
            critic_loss_total += c_loss.item()

            # ── Delayed actor update ───────────────────────────────
            if self._critic_steps % MATD3_POLICY_DELAY == 0:
                curr_acts_list = []
                for j in range(N):
                    if j == i:
                        curr_acts_list.append(
                            self.actors[j % len(self.actors)](obs[:, j, :]))
                    else:
                        curr_acts_list.append(actions[:, j, :].detach())
                curr_acts_all = torch.cat(curr_acts_list, dim=-1)

                a_loss = -self.critics[i].q1_only(state, curr_acts_all).mean()
                self.actor_opts[i].zero_grad()
                a_loss.backward()
                nn.utils.clip_grad_norm_(self.actors[i].parameters(),
                                         MAPPO_MAX_GRAD_NORM)
                self.actor_opts[i].step()
                actor_loss_total += a_loss.item()

                # Polyak update targets
                for p, tp in zip(self.actors[i].parameters(),
                                  self.target_actors[i].parameters()):
                    tp.data.copy_(MATD3_TAU * p.data + (1 - MATD3_TAU) * tp.data)
                for p, tp in zip(self.critics[i].parameters(),
                                  self.target_critics[i].parameters()):
                    tp.data.copy_(MATD3_TAU * p.data + (1 - MATD3_TAU) * tp.data)

        update_every = MATD3_POLICY_DELAY
        return dict(
            actor_loss  = actor_loss_total  / N,
            critic_loss = critic_loss_total / N,
            entropy     = 0.0,
            approx_kl   = 0.0,
        )

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {'method': self.METHOD, 'total_updates': self.total_updates}
        for i in range(self.n_agents):
            state[f'actor_{i}']         = self.actors[i].state_dict()
            state[f'target_actor_{i}']  = self.target_actors[i].state_dict()
            state[f'critic_{i}']        = self.critics[i].state_dict()
            state[f'target_critic_{i}'] = self.target_critics[i].state_dict()
        torch.save(state, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.total_updates = ckpt.get('total_updates', 0)
        for i in range(self.n_agents):
            self.actors[i].load_state_dict(ckpt[f'actor_{i}'])
            self.target_actors[i].load_state_dict(ckpt[f'target_actor_{i}'])
            self.critics[i].load_state_dict(ckpt[f'critic_{i}'])
            self.target_critics[i].load_state_dict(ckpt[f'target_critic_{i}'])
