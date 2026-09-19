"""
MAPPO Baseline  (Table I — Row 2)
===================================
Standard Multi-Agent PPO **without** diffusion-enhanced offloading.

Differences from DMJO (Row 1):
  • Actor : GaussianOnlyActor  — Gaussian MLP, no denoiser head
  • No auxiliary diffusion loss term in the PPO update
  • Uses the same centralised critic and GAE advantage estimation
  • NOMA pairing and cooperative caching are still active in the *env*,
    but the policy is not diffusion-guided.

The class is a thin wrapper around the existing MAPPOAgent machinery
with USE_DIFFUSION forced to False so the file is self-contained and
does not depend on the config flag.

Interface (identical to MAPPOAgent):
    get_actions(obs_all, state, num_active) → actions, log_probs, values
    get_values(state)                       → values
    update(buffer: RolloutBuffer)           → metrics dict
    save(path) / load(path)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from config import (
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_CLIP_EPS,
    MAPPO_ENTROPY_COEF, MAPPO_VALUE_COEF, MAPPO_MAX_GRAD_NORM,
    MAPPO_PPO_EPOCHS, MAPPO_NUM_MINI_BATCHES,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
)
from src.diffusion import GaussianOnlyActor   # no diffusion head
from src.buffer import RolloutBuffer


# ================================================================
# CENTRALISED CRITIC  (identical to MAPPOAgent's)
# ================================================================
class _CentralisedCritic(nn.Module):
    def __init__(self, state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_agents),
        )

    def forward(self, state):
        return self.net(state)


# ================================================================
# MAPPO BASELINE
# ================================================================
class MAPPOBaseline:
    """
    Multi-Agent PPO without diffusion-enhanced offloading (Row 2).

    Actor  : GaussianOnlyActor (standard Gaussian MLP)
    Critic : CentralisedCritic (identical to DMJO)
    Update : PPO clipped objective + entropy bonus
             (no auxiliary diffusion loss — d_loss term is absent)
    """

    METHOD = "MAPPO"

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device="cpu", share_actor=True):
        self.obs_dim     = obs_dim
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.n_agents    = n_agents
        self.device      = device
        self.share_actor = share_actor

        # ── Actors (Gaussian-only) ───────────────────────────────
        if share_actor:
            self.actor  = GaussianOnlyActor(
                obs_dim, action_dim, hidden_dim=MAPPO_HIDDEN_DIM
            ).to(device)
            self.actors = [self.actor] * n_agents
            actor_params = list(self.actor.parameters())
        else:
            self.actors = [
                GaussianOnlyActor(obs_dim, action_dim,
                                  hidden_dim=MAPPO_HIDDEN_DIM).to(device)
                for _ in range(n_agents)
            ]
            self.actor   = self.actors[0]
            actor_params = []
            for a in self.actors:
                actor_params.extend(list(a.parameters()))

        # ── Centralised Critic ───────────────────────────────────
        self.critic = _CentralisedCritic(
            state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM
        ).to(device)

        self.actor_optimizer  = optim.Adam(actor_params, lr=MAPPO_LR_ACTOR)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=MAPPO_LR_CRITIC)

        self.total_updates = 0

    # ----------------------------------------------------------
    # Action selection
    # ----------------------------------------------------------
    @torch.inference_mode()
    def get_actions(self, obs_all, state, num_active=None):
        n_active  = num_active or self.n_agents
        actions   = np.zeros((self.n_agents, self.action_dim))
        log_probs = np.zeros(self.n_agents)

        if self.share_actor:
            obs_t  = torch.from_numpy(
                np.ascontiguousarray(obs_all[:n_active], dtype=np.float32)
            ).to(self.device, non_blocking=True)
            a, lp  = self.actor.get_action_and_log_prob(obs_t)
            actions[:n_active]   = a.cpu().numpy()
            log_probs[:n_active] = lp.cpu().numpy().flatten()
        else:
            for i in range(n_active):
                obs_t = torch.from_numpy(
                    np.ascontiguousarray(obs_all[i], dtype=np.float32)
                ).unsqueeze(0).to(self.device, non_blocking=True)
                a, lp        = self.actors[i % len(self.actors)].get_action_and_log_prob(obs_t)
                actions[i]   = a.cpu().numpy().flatten()
                log_probs[i] = lp.cpu().numpy().item()

        state_t = torch.from_numpy(
            np.ascontiguousarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        values = self.critic(state_t).cpu().numpy().flatten()

        return actions, log_probs, values

    @torch.inference_mode()
    def get_values(self, state):
        state_t = torch.from_numpy(
            np.ascontiguousarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        return self.critic(state_t).cpu().numpy().flatten()

    # ----------------------------------------------------------
    # PPO Update (no diffusion loss)
    # ----------------------------------------------------------
    def update(self, buffer: RolloutBuffer) -> dict:
        self.total_updates += 1
        metrics   = dict(actor_loss=0.0, critic_loss=0.0,
                         entropy=0.0, approx_kl=0.0)
        n_updates = 0
        KL_TARGET = 0.03
        kl_exceeded = False

        actors_to_train = [self.actor] if self.share_actor else self.actors

        for epoch in range(MAPPO_PPO_EPOCHS):
            if kl_exceeded:
                break
            for batch in buffer.get_batches(MAPPO_NUM_MINI_BATCHES):
                obs         = batch['obs'].to(self.device, non_blocking=True)
                states      = batch['states'].to(self.device, non_blocking=True)
                actions     = batch['actions'].to(self.device, non_blocking=True)
                old_lp      = batch['old_log_probs'].to(self.device, non_blocking=True)
                advantages  = batch['advantages'].to(self.device, non_blocking=True)
                returns     = batch['returns'].to(self.device, non_blocking=True)
                old_values  = batch['old_values'].to(self.device, non_blocking=True)

                adv_mean   = advantages.mean()
                adv_std    = advantages.std().clamp(min=1e-8)
                advantages = (advantages - adv_mean) / adv_std

                # ── Actor ──
                skip = False
                actor_loss_sum = entropy_sum = 0.0
                last_ratio     = None

                for actor in actors_to_train:
                    actor.train()
                    new_lp, entropy = actor.evaluate_actions(obs, actions)
                    if torch.isnan(new_lp).any() or torch.isinf(new_lp).any():
                        skip = True
                        break

                    ratio  = torch.exp(torch.clamp(new_lp - old_lp, -5.0, 5.0))
                    surr1  = ratio * advantages
                    surr2  = torch.clamp(ratio,
                                         1 - MAPPO_CLIP_EPS,
                                         1 + MAPPO_CLIP_EPS) * advantages
                    a_loss = -torch.min(surr1, surr2).mean()
                    e_loss = -entropy.mean()

                    # NOTE: no d_loss term here (MAPPO baseline)
                    total  = a_loss + MAPPO_ENTROPY_COEF * e_loss
                    if total.abs().item() > 1e4:
                        skip = True
                        break

                    self.actor_optimizer.zero_grad()
                    total.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(),
                                             MAPPO_MAX_GRAD_NORM)
                    self.actor_optimizer.step()

                    actor_loss_sum += a_loss.item()
                    entropy_sum    += (-e_loss).item()
                    last_ratio      = ratio.detach()

                if skip:
                    continue

                n_actors    = len(actors_to_train)
                last_ratio  = last_ratio

                # ── Critic ──
                self.critic.train()
                agent_ids   = batch['agent_ids'].to(self.device)
                vals_all    = self.critic(states)
                vals_pred   = vals_all[
                    torch.arange(vals_all.size(0), device=self.device),
                    agent_ids]
                vals_clip   = old_values + torch.clamp(
                    vals_pred - old_values, -MAPPO_CLIP_EPS, MAPPO_CLIP_EPS)
                c_loss      = MAPPO_VALUE_COEF * torch.max(
                    F.mse_loss(vals_pred, returns),
                    F.mse_loss(vals_clip, returns))

                self.critic_optimizer.zero_grad()
                c_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),
                                          MAPPO_MAX_GRAD_NORM)
                self.critic_optimizer.step()

                with torch.no_grad():
                    approx_kl = ((last_ratio - 1) - last_ratio.log()).mean()

                if approx_kl.item() > KL_TARGET:
                    kl_exceeded = True

                metrics['actor_loss']  += actor_loss_sum / n_actors
                metrics['critic_loss'] += c_loss.item()
                metrics['entropy']     += entropy_sum / n_actors
                metrics['approx_kl']   += approx_kl.item()
                n_updates += 1

                if kl_exceeded:
                    break

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        return metrics

    # ----------------------------------------------------------
    # Checkpoint
    # ----------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'method':        self.METHOD,
            'critic':        self.critic.state_dict(),
            'critic_opt':    self.critic_optimizer.state_dict(),
            'actor_opt':     self.actor_optimizer.state_dict(),
            'total_updates': self.total_updates,
        }
        if self.share_actor:
            state['actor'] = self.actor.state_dict()
        else:
            for i, a in enumerate(self.actors):
                state[f'actor_{i}'] = a.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.actor_optimizer.load_state_dict(ckpt['actor_opt'])
        self.total_updates = ckpt.get('total_updates', 0)
        if self.share_actor:
            self.actor.load_state_dict(ckpt['actor'])
        else:
            for i, a in enumerate(self.actors):
                a.load_state_dict(ckpt[f'actor_{i}'])
