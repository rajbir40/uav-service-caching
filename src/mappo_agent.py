"""
Diffusion-MAPPO Agent
=======================
Multi-Agent PPO with:
  - Shared Diffusion Actor for decentralised execution (Paper §IV.E)
  - Centralised Critic with global state (CTDE paradigm)
  - GAE advantage estimation
  - PPO clipped objective with KL early stopping and entropy bonus
  - Auxiliary diffusion denoising loss

Actor class is selected by config.USE_DIFFUSION:
  True  → DiffusionActor      (full model, ablation rows 1, 2, 4, 5)
  False → GaussianOnlyActor   (ablation rows 3 and 6)
Both classes expose identical public methods.

References:
  Yu et al. (2022) "The Surprising Effectiveness of PPO in MARL"
  Paper §IV.D–IV.E
"""

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_CLIP_EPS,
    MAPPO_ENTROPY_COEF, MAPPO_VALUE_COEF, MAPPO_MAX_GRAD_NORM,
    MAPPO_PPO_EPOCHS, MAPPO_NUM_MINI_BATCHES,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
    DIFFUSION_DENOISING_STEPS,
    USE_DIFFUSION,
)
from src.diffusion import DiffusionActor, GaussianOnlyActor
from src.buffer import RolloutBuffer

_ActorClass = DiffusionActor if USE_DIFFUSION else GaussianOnlyActor


class CentralisedCritic(nn.Module):
    """
    Centralised critic for CTDE: global state → V(s) per agent.
    Shared across all agents (parameter sharing).
    """
    def __init__(self, state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, n_agents),
        )

    def forward(self, state):
        return self.net(state)


class MAPPOAgent:
    """
    Diffusion-MAPPO agent.

    share_actor=True (default): one shared actor for all UAVs —
      better sample efficiency; per-agent heterogeneity is captured
      through different local observations.
    share_actor=False: independent actor per agent.
    """

    METHOD = "DMJO"   # must match the key used in agent_factory.py / train.py

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device='cpu', share_actor=True):
        self.obs_dim     = obs_dim
        self.state_dim   = state_dim
        self.action_dim  = action_dim
        self.n_agents    = n_agents
        self.device      = device
        self.share_actor = share_actor

        if share_actor:
            self.actor  = _ActorClass(
                obs_dim, action_dim,
                hidden_dim=MAPPO_HIDDEN_DIM,
                K=DIFFUSION_DENOISING_STEPS,
            ).to(device)
            self.actors    = [self.actor] * n_agents
            actor_params   = list(self.actor.parameters())
        else:
            self.actors = [
                _ActorClass(
                    obs_dim, action_dim,
                    hidden_dim=MAPPO_HIDDEN_DIM,
                    K=DIFFUSION_DENOISING_STEPS,
                ).to(device)
                for _ in range(n_agents)
            ]
            self.actor   = self.actors[0]
            actor_params = []
            for a in self.actors:
                actor_params.extend(list(a.parameters()))

        self.critic = CentralisedCritic(
            state_dim, n_agents, hidden_dim=MAPPO_CRITIC_HIDDEN_DIM
        ).to(device)

        # Bug fix: when share_actor=False each actor needs its own optimizer so
        # that zero_grad / step calls do not cross-contaminate gradient accumulation
        # across actors.  When share_actor=True we keep a single optimizer (same
        # behaviour as before).
        if share_actor:
            self.actor_optimizer  = optim.Adam(actor_params, lr=MAPPO_LR_ACTOR)
            self.actor_optimizers = [self.actor_optimizer]
        else:
            self.actor_optimizers = [
                optim.Adam(list(a.parameters()), lr=MAPPO_LR_ACTOR)
                for a in self.actors
            ]
            self.actor_optimizer = self.actor_optimizers[0]   # kept for save/load compat

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=MAPPO_LR_CRITIC)

        self.total_updates = 0

    @torch.inference_mode()
    def get_actions(self, obs_all, state, num_active=None):
        """
        Select actions for all agents in one batched forward pass.

        Args:
            obs_all    : np.ndarray (n_agents, obs_dim)
            state      : np.ndarray (state_dim,)
            num_active : number of active agents (inactive slots stay zero)

        Returns:
            actions   : np.ndarray (n_agents, action_dim)
            log_probs : np.ndarray (n_agents,)
            values    : np.ndarray (n_agents,)
        """
        n_active  = num_active or self.n_agents
        actions   = np.zeros((self.n_agents, self.action_dim))
        log_probs = np.zeros(self.n_agents)

        if self.share_actor:
            obs_batch = torch.from_numpy(
                np.ascontiguousarray(obs_all[:n_active], dtype=np.float32)
            ).to(self.device, non_blocking=True)
            a_batch, lp_batch    = self.actor.get_action_and_log_prob(obs_batch)
            actions[:n_active]   = a_batch.cpu().numpy()
            log_probs[:n_active] = lp_batch.cpu().numpy().flatten()
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
        """Return V(s) from the centralised critic."""
        state_t = torch.from_numpy(
            np.ascontiguousarray(state, dtype=np.float32)
        ).unsqueeze(0).to(self.device, non_blocking=True)
        return self.critic(state_t).cpu().numpy().flatten()

    def update(self, buffer: RolloutBuffer) -> dict:
        """
        Run PPO update over one rollout buffer.
        Returns a dict of averaged training metrics.
        """
        self.total_updates += 1
        metrics   = dict(actor_loss=0.0, critic_loss=0.0, entropy=0.0,
                         diffusion_loss=0.0, approx_kl=0.0)
        n_updates = 0
        KL_TARGET = 0.03
        kl_exceeded = False

        actors_to_train = [self.actor] if self.share_actor else self.actors

        for _ in range(MAPPO_PPO_EPOCHS):
            if kl_exceeded:
                break

            for batch in buffer.get_batches(MAPPO_NUM_MINI_BATCHES):
                obs           = batch['obs'].to(self.device, non_blocking=True)
                states        = batch['states'].to(self.device, non_blocking=True)
                actions       = batch['actions'].to(self.device, non_blocking=True)
                old_log_probs = batch['old_log_probs'].to(self.device, non_blocking=True)
                advantages    = batch['advantages'].to(self.device, non_blocking=True)
                returns       = batch['returns'].to(self.device, non_blocking=True)
                old_values    = batch['old_values'].to(self.device, non_blocking=True)
                agent_ids     = batch['agent_ids'].to(self.device)

                advantages = (advantages - advantages.mean()) / (advantages.std().clamp(min=1e-8))

                # Actor update
                actor_loss_sum, entropy_sum, diff_loss_sum = 0.0, 0.0, 0.0
                skip_batch = False
                # Bug fix: accumulate ratios from ALL actors for a correct KL
                # estimate rather than keeping only the last actor's ratio.
                ratio_sum  = None

                for actor, opt in zip(actors_to_train, self.actor_optimizers):
                    actor.train()
                    new_log_probs, entropy = actor.evaluate_actions(obs, actions)

                    if torch.isnan(new_log_probs).any() or torch.isinf(new_log_probs).any():
                        skip_batch = True
                        break

                    ratio  = torch.exp(torch.clamp(new_log_probs - old_log_probs, -5.0, 5.0))
                    surr1  = ratio * advantages
                    surr2  = torch.clamp(ratio, 1 - MAPPO_CLIP_EPS, 1 + MAPPO_CLIP_EPS) * advantages
                    a_loss = -torch.min(surr1, surr2).mean()
                    d_loss = actor.compute_diffusion_loss(obs, actions)

                    total_loss = a_loss + MAPPO_ENTROPY_COEF * (-entropy.mean()) + 0.1 * d_loss

                    if total_loss.abs().item() > 1e4:
                        skip_batch = True
                        break

                    # Bug fix: use this actor's own optimizer so zero_grad / step
                    # only touches its parameters, not those of sibling actors.
                    opt.zero_grad()
                    total_loss.backward()
                    nn.utils.clip_grad_norm_(actor.parameters(), MAPPO_MAX_GRAD_NORM)
                    opt.step()

                    actor_loss_sum += a_loss.item()
                    entropy_sum    += entropy.mean().item()
                    diff_loss_sum  += d_loss.item()
                    ratio_sum       = (ratio.detach() if ratio_sum is None
                                       else ratio_sum + ratio.detach())

                if skip_batch:
                    continue

                n_a = len(actors_to_train)

                # Critic update
                self.critic.train()
                values_all    = self.critic(states)
                values_pred   = values_all[
                    torch.arange(values_all.size(0), device=self.device), agent_ids]
                values_clipped = old_values + torch.clamp(
                    values_pred - old_values, -MAPPO_CLIP_EPS, MAPPO_CLIP_EPS)
                critic_loss = MAPPO_VALUE_COEF * torch.max(
                    F.mse_loss(values_pred, returns),
                    F.mse_loss(values_clipped, returns))

                self.critic_optimizer.zero_grad()
                critic_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), MAPPO_MAX_GRAD_NORM)
                self.critic_optimizer.step()

                with torch.no_grad():
                    # Bug fix: average ratio across all actors before computing KL
                    # so early-stopping reflects the joint policy shift, not just
                    # the last actor updated in the loop.
                    mean_ratio = ratio_sum / len(actors_to_train)
                    approx_kl  = ((mean_ratio - 1) - mean_ratio.log()).mean()

                if approx_kl.item() > KL_TARGET:
                    kl_exceeded = True

                metrics['actor_loss']     += actor_loss_sum / n_a
                metrics['critic_loss']    += critic_loss.item()
                metrics['entropy']        += entropy_sum / n_a
                metrics['diffusion_loss'] += diff_loss_sum / n_a
                metrics['approx_kl']      += approx_kl.item()
                n_updates += 1

                if kl_exceeded:
                    break

        for k in metrics:
            metrics[k] /= max(n_updates, 1)
        return metrics

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state = {
            'critic':        self.critic.state_dict(),
            'critic_opt':    self.critic_optimizer.state_dict(),
            'actor_opt':     self.actor_optimizers[0].state_dict(),   # compat alias
            'total_updates': self.total_updates,
            'ablation_mode': __import__('config').ABLATION_MODE,
        }
        if self.share_actor:
            state['actor'] = self.actor.state_dict()
        else:
            for i, (a, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                state[f'actor_{i}']     = a.state_dict()
                state[f'actor_opt_{i}'] = opt.state_dict()
        torch.save(state, path)

    def load(self, path: str):
        ckpt = torch.load(path, map_location=self.device)
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_optimizer.load_state_dict(ckpt['critic_opt'])
        self.total_updates = ckpt.get('total_updates', 0)
        if self.share_actor:
            self.actor.load_state_dict(ckpt['actor'])
            self.actor_optimizers[0].load_state_dict(ckpt['actor_opt'])
        else:
            for i, (a, opt) in enumerate(zip(self.actors, self.actor_optimizers)):
                a.load_state_dict(ckpt[f'actor_{i}'])
                # Graceful fallback: checkpoints saved before this fix only have
                # 'actor_opt' (the alias for actor 0); load it for actor 0 and
                # leave the rest at fresh-init state.
                key = f'actor_opt_{i}' if f'actor_opt_{i}' in ckpt else ('actor_opt' if i == 0 else None)
                if key:
                    opt.load_state_dict(ckpt[key])