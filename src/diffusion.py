"""
Diffusion-Gaussian Hybrid Actor for MAPPO
============================================
Dual-head architecture (Paper §IV.D):

  HEAD 1 — Gaussian MLP
    obs → μ, σ → Normal(μ, σ) → differentiable log_prob and entropy for PPO.

  HEAD 2 — Diffusion Denoiser
    obs → reverse VP-SDE → structured action proposals via iterative denoising.
    Trained via noise-prediction MSE. Used for rollout sampling.

  Both heads share the same obs encoder so they co-evolve during training.

GaussianOnlyActor: drop-in replacement removing the diffusion head.
  Used by ablation rows 3 (w/o Diffusion) and 6 (Minimal).
  Exposes identical public methods so MAPPOAgent needs no conditional logic.

References:
  Paper §IV.D; [P6] Liang et al., Section IV-C
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (DIFFUSION_DENOISING_STEPS, DIFFUSION_BETA_A,
                    DIFFUSION_BETA_B, DIFFUSION_HIDDEN_DIM)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal timestep embedding for the denoising network."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half = self.dim // 2
        emb = math.log(10000) / (half - 1)
        emb = torch.exp(torch.arange(half, device=device) * -emb)
        emb = x[:, None].float() * emb[None, :]
        return torch.cat([emb.sin(), emb.cos()], dim=-1)


class VPNoiseSchedule:
    """VP-SDE noise schedule with cosine-shaped beta sequence."""
    def __init__(self, K=DIFFUSION_DENOISING_STEPS,
                 beta_a=DIFFUSION_BETA_A, beta_b=DIFFUSION_BETA_B):
        self.K = K
        betas = []
        for k in range(1, K + 1):
            exp_arg = -beta_a / K - (2*k - 1) / (2*K**2) * (beta_b - beta_a)
            betas.append(np.clip(1.0 - np.exp(exp_arg), 1e-4, 0.999))
        self.betas          = np.array(betas)
        self.alphas         = 1.0 - self.betas
        self.alpha_bars     = np.cumprod(self.alphas)
        self.alpha_bars_prev = np.concatenate([[1.0], self.alpha_bars[:-1]])
        self.beta_tildes    = ((1 - self.alpha_bars_prev) /
                               (1 - self.alpha_bars) * self.betas)


class ObsEncoder(nn.Module):
    """Shared feature extractor — gradients from both heads flow through this."""
    def __init__(self, obs_dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, obs):
        obs = torch.nan_to_num(obs, nan=0.0, posinf=1e3, neginf=-1e3)
        return self.net(obs)


class GaussianHead(nn.Module):
    """
    Gaussian policy head: obs_features → (μ, σ) → Normal.
    Provides differentiable log_prob and entropy for PPO.
    """
    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.mu_net = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.SiLU(),
            nn.Linear(feature_dim // 2, action_dim),
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim) - 0.5)

    def forward(self, features):
        mu  = torch.clamp(self.mu_net(features), -10.0, 10.0)
        mu  = torch.nan_to_num(mu, nan=0.0, posinf=10.0, neginf=-10.0)
        std = torch.exp(torch.clamp(self.log_std, -5, 2)).expand_as(mu)
        return mu, std

    def get_distribution(self, features):
        mu, std = self.forward(features)
        return torch.distributions.Normal(mu, std)

    def sample(self, features):
        dist    = self.get_distribution(features)
        raw     = dist.rsample()
        action  = torch.tanh(raw)
        log_prob = dist.log_prob(raw) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1)

    def evaluate(self, features, actions):
        """Log_prob and entropy for given actions (differentiable)."""
        dist     = self.get_distribution(features)
        raw      = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        log_prob = dist.log_prob(raw) - torch.log(1 - actions.pow(2) + 1e-6)
        entropy  = dist.entropy().sum(dim=-1)
        return log_prob.sum(dim=-1), entropy


class DenoisingNetwork(nn.Module):
    """Predicts noise ε given (x_k, timestep k, obs_features). Paper §IV.D Eq.(26)"""
    def __init__(self, feature_dim, action_dim, time_emb_dim=32):
        super().__init__()
        self.time_emb = SinusoidalPosEmb(time_emb_dim)
        input_dim = action_dim + feature_dim + time_emb_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, feature_dim),
            nn.SiLU(),
            nn.Linear(feature_dim, action_dim),
        )

    def forward(self, x_k, k, features):
        return self.net(torch.cat([x_k, features, self.time_emb(k)], dim=-1))


class DiffusionActor(nn.Module):
    """
    Dual-head actor for Diffusion-MAPPO (Paper §IV.D).

    Rollout (get_action_and_log_prob):
      Blends diffusion and Gaussian samples; scores action with Gaussian head
      for PPO log_prob.

    PPO update (evaluate_actions):
      Gaussian head provides differentiable log_prob and entropy.
      Diffusion head trained in parallel via noise-prediction MSE.

    anneal_blend() linearly shifts from 70% to 30% diffusion actions over
    training as the Gaussian head improves.
    """

    def __init__(self, obs_dim, action_dim,
                 hidden_dim=DIFFUSION_HIDDEN_DIM,
                 K=DIFFUSION_DENOISING_STEPS,
                 beta_a=DIFFUSION_BETA_A,
                 beta_b=DIFFUSION_BETA_B):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.K          = K

        self.encoder       = ObsEncoder(obs_dim, hidden_dim)
        self.gaussian_head = GaussianHead(hidden_dim, action_dim)

        self.schedule  = VPNoiseSchedule(K, beta_a, beta_b)
        self.denoiser  = DenoisingNetwork(hidden_dim, action_dim)

        self.register_buffer('betas',       torch.FloatTensor(self.schedule.betas))
        self.register_buffer('alphas',      torch.FloatTensor(self.schedule.alphas))
        self.register_buffer('alpha_bars',  torch.FloatTensor(self.schedule.alpha_bars))
        self.register_buffer('beta_tildes', torch.FloatTensor(self.schedule.beta_tildes))

        self.diffusion_blend = 0.7

    def _diffusion_sample(self, features):
        """Generate action via reverse diffusion."""
        batch  = features.shape[0]
        device = features.device
        x      = torch.randn(batch, self.action_dim, device=device)

        for k in reversed(range(self.K)):
            kt          = torch.full((batch,), k, device=device, dtype=torch.long)
            eps_pred    = self.denoiser(x, kt, features)
            alpha_k     = self.alphas[k]
            alpha_bar_k = self.alpha_bars[k]
            beta_k      = self.betas[k]
            coef1       = 1.0 / torch.sqrt(alpha_k)
            coef2       = beta_k / torch.sqrt(1.0 - alpha_bar_k)
            mean        = coef1 * (x - coef2 * eps_pred)
            x = mean + torch.sqrt(self.beta_tildes[k]) * torch.randn_like(x) if k > 0 else mean

        return torch.tanh(x)

    def forward(self, obs):
        return self._diffusion_sample(self.encoder(obs))

    def get_action_and_log_prob(self, obs):
        """Blend diffusion and Gaussian actions; score with Gaussian head."""
        features = self.encoder(obs)

        with torch.no_grad():
            diff_action = self._diffusion_sample(features)

        gauss_action, _ = self.gaussian_head.sample(features)

        use_diffusion = (torch.rand(obs.shape[0], 1, device=obs.device)
                         < self.diffusion_blend)
        action   = torch.where(use_diffusion, diff_action, gauss_action)
        log_prob, _ = self.gaussian_head.evaluate(features.detach(), action)
        return action, log_prob

    def evaluate_actions(self, obs, actions):
        """Differentiable log_prob and entropy for PPO update."""
        features = self.encoder(obs)
        return self.gaussian_head.evaluate(features, actions)

    def compute_diffusion_loss(self, obs, actions):
        """Noise-prediction MSE to keep the denoiser aligned with the policy."""
        features    = self.encoder(obs)
        batch       = features.shape[0]
        device      = features.device
        actions_raw = torch.atanh(torch.clamp(actions, -0.999, 0.999))
        k           = torch.randint(0, self.K, (batch,), device=device)
        alpha_bar   = self.alpha_bars[k].unsqueeze(-1)
        noise       = torch.randn_like(actions_raw)
        x_k         = (torch.sqrt(alpha_bar) * actions_raw +
                       torch.sqrt(1 - alpha_bar) * noise)
        return F.mse_loss(self.denoiser(x_k, k, features), noise)

    def anneal_blend(self, progress):
        """Linearly anneal diffusion blend ratio from 0.7 → 0.3 over training."""
        self.diffusion_blend = max(0.3, 0.7 - 0.4 * progress)


class GaussianOnlyActor(nn.Module):
    """
    Standard Gaussian MLP actor — no diffusion head.
    Drop-in replacement for DiffusionActor (ablation rows 3 and 6).
    Exposes identical public methods; compute_diffusion_loss returns zero.
    """

    def __init__(self, obs_dim, action_dim,
                 hidden_dim=DIFFUSION_HIDDEN_DIM, **kwargs):
        super().__init__()
        self.obs_dim    = obs_dim
        self.action_dim = action_dim
        self.encoder       = ObsEncoder(obs_dim, hidden_dim)
        self.gaussian_head = GaussianHead(hidden_dim, action_dim)

    def forward(self, obs):
        return self.gaussian_head.sample(self.encoder(obs))[0]

    def get_action_and_log_prob(self, obs):
        features = self.encoder(obs)
        return self.gaussian_head.sample(features)

    def evaluate_actions(self, obs, actions):
        return self.gaussian_head.evaluate(self.encoder(obs), actions)

    def compute_diffusion_loss(self, obs, actions):
        return torch.tensor(0.0, device=obs.device, requires_grad=True)

    def anneal_blend(self, progress):
        pass
