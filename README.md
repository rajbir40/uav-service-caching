# Diffusion-DRL-MultiUAV

**DMJO — Diffusion-Enhanced Multi-Agent Joint Optimization for UAV-Assisted Mobile Edge Computing**

A multi-UAV Mobile Edge Computing (MEC) framework that jointly optimizes UAV trajectory control, cooperative task offloading, service caching, and NOMA-based uplink scheduling using a Diffusion-enhanced Multi-Agent Proximal Policy Optimization (MAPPO) algorithm.

All the Results and Plot Are Available At Google Drive Link : https://drive.google.com/drive/folders/1DrcdYP7IQtSxFBGbz56U5fJvNVmogn-U?usp=sharing

---

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Key Features](#key-features)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Training](#training)
- [Evaluation](#evaluation)
- [Ablation Study](#ablation-study)
- [Baseline Comparisons](#baseline-comparisons)
- [Generating Figures](#generating-figures)
- [Configuration Reference](#configuration-reference)
- [Environment Details](#environment-details)
- [Output Structure](#output-structure)
- [License](#license)

---

## Overview

DMJO addresses the joint optimization problem in jammer-aware multi-UAV MEC-IoT networks. The core framework combines:

- A **dual-head diffusion actor** that uses iterative reverse VP-SDE denoising for high-quality action proposals alongside a Gaussian PPO head for differentiable policy updates
- A **centralised critic** over the global system state under the CTDE (Centralised Training, Decentralised Execution) paradigm
- A **3-tier cooperative caching** system (local UAV cache → A2A peer fetch → Base Station backhaul) governed by a Frequency+Weight MAB strategy
- A **two-timescale control** architecture separating per-slot trajectory decisions from macro-interval cache update decisions
- **NOMA uplink scheduling** with optimal power splitting via bisection to serve two IoT devices per UAV per slot
- **Domain randomisation** over UAV count, IoT count, jammer configurations, area radius, and wind speed for robust generalisation

**Objective:** Minimise a weighted sum of end-to-end task latency and UAV propulsion energy across the deployment:

```
min Σ_t ( α · L_total(t) + β · E_UAV(t) )
```

with default weights α = 0.6 (latency) and β = 0.4 (energy).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     IoT Devices (up to 180)                 │
│         Gauss-Markov mobility · variable TX power           │
└────────────────────┬────────────────────────────────────────┘
                     │  NOMA Uplink (A2G channel, Rician fading)
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              UAV Cluster (3 – 8 heterogeneous UAVs)         │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐ │
│  │  Diffusion  │  │  3-Tier      │  │  NOMA Scheduler    │ │
│  │  MAPPO      │  │  Coop Cache  │  │  (bisection α-     │ │
│  │  Actor      │  │  MAB policy  │  │   split + SIC)     │ │
│  └─────────────┘  └──────────────┘  └────────────────────┘ │
│         ↕ A2A cooperative cache transfer (5 MHz)            │
└────────────────────┬────────────────────────────────────────┘
                     │  Backhaul fallback
                     ▼
              Base Station (BS)
```

**Two-timescale control loop:**

| Timescale | Period | Decision |
|---|---|---|
| Fast (per slot) | 1 slot | UAV trajectory, NOMA pairing, task offloading |
| Slow (macro) | K_C = 25 slots | Cache content refresh via MAB |

---

## Key Features

**Diffusion-Enhanced Actor**
- Dual-head architecture: shared `ObsEncoder` → Gaussian head (PPO log_prob / entropy) + Diffusion denoiser head (action proposals)
- VP-SDE noise schedule with cosine-shaped beta sequence over K = 3 reverse steps
- Sinusoidal timestep embeddings for the denoising network
- `GaussianOnlyActor` drop-in replacement for ablation experiments

**Heterogeneous UAV Fleet**
- 8 UAV capability profiles (CPU 1.2 – 3.2 GHz, cache 60 – 160 MB, bandwidth 8 – 16 MHz, altitude 85 – 115 m)
- Rotary-wing propulsion energy model with wind drag
- Ornstein-Uhlenbeck wind process (0 – 8 m/s) affecting both propulsion energy and LoS channel quality

**Realistic Channel Model**
- Probabilistic A2G LoS/NLoS with elevation-angle-dependent probability
- Rician fading with K-factor = 10
- Wind-induced antenna jitter degrading LoS probability
- Jammer interference folded directly into effective noise PSD (interference radius 120 m, penalty radius 180 m)

**Cooperative 3-Tier Caching**
- Tier 1: Local UAV cache hit (zero extra delay)
- Tier 2: A2A cooperative peer fetch (0.05 s)
- Tier 3: Base station backhaul (0.40 s)
- Frequency+Weight priority scoring: S_l = 0.6·(freq_score) + 0.4·(weight_score)
- Greedy knapsack placement under per-UAV capacity; diversity-aware initialisation

**Domain Randomisation**
- UAV count: 3 – 8 · IoT count: 80 – 150 · Jammer count: 1 – 3
- Jammer TX power: 15 – 25 dBm · Area radius: 400 – 600 m
- IoT TX power: 15 – 23 dBm · Noise PSD: −135 – −125 dBm/Hz
- Wind speed: 0 – 5 m/s · Jammer interference radius: 80 – 160 m

---

## Repository Structure

```
Diffusion-DRL-MultiUAV/
├── config.py              # All hyperparameters and environment constants
├── train.py               # Training entry point (all 6 methods)
├── test_custom.py         # Evaluation on custom environment configurations
├── plot_ieee.py           # Generates all evaluation figures (P1–P11)
├── requirements.txt
└── src/
    ├── env.py             # Multi-UAV MEC environment (CTDE, NOMA, jamming, wind)
    ├── diffusion.py       # DiffusionActor + GaussianOnlyActor + VP-SDE schedule
    ├── mappo_agent.py     # MAPPO with centralised critic (DMJO main agent)
    ├── caching.py         # 3-tier cooperative cache manager (MAB strategy)
    ├── channel_model.py   # A2G LoS/NLoS, Rician fading, NOMA communication
    ├── buffer.py          # Rollout buffer with GAE advantage estimation
    ├── metrics.py         # Latency, energy, cache hit, cost tracking
    ├── logger.py          # Training logger, trajectory recorder, parameter sweeps
    ├── agent_factory.py   # Single-point model switcher for all 6 methods
    └── baselines/
        ├── mappo_baseline.py  # MAPPO without diffusion
        ├── maddpg_agent.py    # Deterministic multi-agent AC
        ├── matd3_agent.py     # Double-Q deterministic MARL
        ├── greedy_agent.py    # Nearest-UAV greedy (no learning)
        └── local_agent.py     # All tasks local (no learning)
```

---

## Requirements

```
Python    >= 3.10
PyTorch   >= 2.0.0
numpy     >= 1.24.0
matplotlib >= 3.7.0
imageio   >= 2.31.0
imageio-ffmpeg >= 0.4.8
```

GPU (CUDA) is strongly recommended. CPU training is supported but significantly slower.

---

## Installation

```bash
git clone https://github.com/your-username/Diffusion-DRL-MultiUAV.git
cd Diffusion-DRL-MultiUAV
pip install -r requirements.txt
```

---

## Quick Start

**Train the full DMJO model:**
```bash
python train.py --device cuda
```

**Evaluate a trained model:**
```bash
python test_custom.py --device cuda
```

**Generate all evaluation figures:**
```bash
python plot_ieee.py
```

---

## Training

### Full DMJO Model

```bash
python train.py --device cuda
```

### All Comparison Methods

```bash
python train.py --method DMJO             # Full diffusion-enhanced model (default)
python train.py --method MAPPO            # MAPPO without diffusion
python train.py --method MADDPG           # Deterministic multi-agent AC
python train.py --method MATD3            # Double-Q deterministic MARL
python train.py --method GreedyOffloading # Nearest-UAV greedy baseline
python train.py --method LocalExecution   # All tasks local baseline
```

All method flags combine with all other CLI arguments:
```bash
python train.py --method MATD3 --total-steps 5000000 --device cuda
```

### CLI Arguments

| Argument | Default | Description |
|---|---|---|
| `--method` | `DMJO` | Algorithm: `DMJO`, `MAPPO`, `MADDPG`, `MATD3`, `GreedyOffloading`, `LocalExecution` |
| `--device` | `cpu` | Compute device: `cpu`, `cuda`, or `mps` |
| `--total-steps` | `5000000` | Total environment interaction steps |
| `--seed` | `42` | Random seed for reproducibility |
| `--no-training-video` | off | Skip mid-training 3D video rendering (speeds up training) |
| `--ablation` | `full` | Ablation variant (see [Ablation Study](#ablation-study)) |
| `--checkpoint` | — | Path to a checkpoint to resume from |
| `--eval` | off | Run evaluation only (requires `--checkpoint`) |
| `--video-frames` | `960` | Number of frames for generated videos |
| `--video-fps` | `24` | Frames per second for generated videos |

### Disabling Mid-Training Videos

Periodic 3D videos are rendered every `VIDEO_INTERVAL` steps by default. To skip them and speed up training, either:

```bash
python train.py --no-training-video
```

or set `ENABLE_TRAINING_VIDEO = False` in `config.py`.

Trajectory plots (2D + 3D) are still generated at every interval. Final evaluation videos after training are never skipped.

---

## Evaluation

Evaluate a trained model on the default or custom environment configuration:

```bash
# Auto-select best checkpoint and use default environment
python test_custom.py

# Custom environment parameters
python test_custom.py --num-iot 120 --num-uavs 6 --num-jammers 2

# Full custom test with multiple episodes
python test_custom.py \
    --num-iot 130 --num-uavs 7 --num-jammers 3 \
    --area-radius 550 --episodes 20

# Manually specify a checkpoint
python test_custom.py \
    --checkpoint results/checkpoints/DMJO/best_model.pt \
    --num-uavs 4 --num-jammers 1

# Run on GPU
python test_custom.py --device cuda --num-uavs 6
```

**Checkpoint auto-selection priority:**
1. `results/checkpoints/best_model.pt` — highest reward during training
2. `results/checkpoints/final_model.pt` — saved at end of training
3. `results/checkpoints/model_step*.pt` — highest step number found

**Evaluation outputs** (saved to `results/test_results/<checkpoint>_<timestamp>/`):

```
logs/
    training_metrics.jsonl  — per-step metrics
    eval_sweeps.json        — parameter sweep results for P2–P11
trajectories/
    flight_path_0000001.npz — UAV + IoT trajectory data
plots/
    P1a_reward.*            — reward convergence curve
    P1b_latency.*           — latency convergence curve
    P2_latency_vs_N.*       — latency vs number of IoT devices
    P3_latency_vs_M.*       — latency vs number of UAVs
    P4_latency_vs_J.*       — latency vs number of jammers
    P5_energy.*             — cumulative UAV energy
    P6_latency_vs_cache.*   — latency vs cache capacity
    P7_cache_vs_zipf.*      — cache hit rate vs Zipf exponent
    P8_traj_2d.*            — 2D UAV trajectory plot
    P8_traj_3d.*            — 3D UAV trajectory plot
    P9_latency_vs_jammer.*  — latency vs jammer transmit power
    P10_latency_vs_task.*   — latency vs task data size
    P11_latency_vs_tx.*     — latency vs IoT TX power
    summary.png             — 4-panel episode metric summary
metrics.json                — summary stats (mean ± std)
episodes.jsonl              — per-episode results
```

---

## Ablation Study

The framework supports six ablation variants to isolate the contribution of each component:

| Mode | Flag | What is changed |
|---|---|---|
| Full DMJO | `full` | Complete framework (default) |
| Single-Timescale | `single_timescale` | Cache updated every slot (K_C = 1 instead of 25) |
| w/o Diffusion | `no_diffusion` | Gaussian-only actor; diffusion denoiser removed |
| w/o MAPPO Trajectory | `no_mappo_traj` | Fixed circular UAV orbits; trajectory learning disabled |
| w/o Cooperative Cache | `no_coop_cache` | Tier-2 A2A fetch disabled; cache misses go directly to BS |
| Minimal | `minimal` | No diffusion and no cooperative caching |

### Run All Ablation Variants

```bash
python train.py --ablation full              --device cuda
python train.py --ablation single_timescale  --device cuda
python train.py --ablation no_diffusion      --device cuda
python train.py --ablation no_mappo_traj     --device cuda
python train.py --ablation no_coop_cache     --device cuda
python train.py --ablation minimal           --device cuda
```

Each variant saves to its own subdirectory under `results/checkpoints/` and `results/logs/` so all runs coexist without overwriting each other.

---

## Baseline Comparisons

All six comparison methods share the same environment and the same agent interface (`get_actions`, `update`, `save`, `load`). Switch between them with the `--method` flag or by editing `METHOD` in `src/agent_factory.py`.

| Method | Type | Description |
|---|---|---|
| `DMJO` | On-policy (ours) | Diffusion-enhanced MAPPO with centralised critic |
| `MAPPO` | On-policy | Standard MAPPO without diffusion denoiser |
| `MADDPG` | Off-policy | Deterministic multi-agent actor-critic |
| `MATD3` | Off-policy | Double-Q deterministic MARL (TD3 extension) |
| `GreedyOffloading` | Rule-based | Nearest-UAV association, no caching awareness |
| `LocalExecution` | Rule-based | All IoT tasks computed locally at the device |

**Off-policy methods (MADDPG, MATD3)** maintain an internal replay buffer. `train.py` automatically calls `agent.push_transition()` after each environment step when an off-policy agent is detected via `is_off_policy(agent)`.

**Rule-based methods (GreedyOffloading, LocalExecution)** have a no-op `update()` and save/load a small JSON metadata file (no model weights).

---

## Generating Figures

After any training run completes, regenerate all evaluation plots with:

```bash
python plot_ieee.py
```

| Figure | Plot file | Description |
|---|---|---|
| P1a | `P1a_reward` | Reward convergence curve |
| P1b | `P1b_latency` | Latency convergence curve |
| P2 | `P2_latency_vs_N` | Average latency vs. number of IoT devices |
| P3 | `P3_latency_vs_M` | Average latency vs. number of UAVs |
| P4 | `P4_latency_vs_J` | Average latency vs. number of jammers |
| P5 | `P5_energy` | Cumulative UAV energy consumption |
| P6 | `P6_latency_vs_cache` | Average latency vs. cache capacity |
| P7 | `P7_cache_vs_zipf` | Cache hit rate vs. Zipf popularity exponent |
| P8 | `P8_traj_2d` / `P8_traj_3d` | 2D and 3D UAV trajectory visualisations |
| P9 | `P9_latency_vs_jammer` | Average latency vs. jammer transmit power |
| P10 | `P10_latency_vs_task` | Average latency vs. task data size |
| P11 | `P11_latency_vs_tx` | Average latency vs. IoT transmit power |

All figures are saved to `results/plots/` in both `.png` and `.pdf` format.

---

## Configuration Reference

All hyperparameters live in `config.py`. The most important groups are listed below.

### Area & Deployment

| Parameter | Default | Description |
|---|---|---|
| `AREA_RADIUS` | 500 m | Deployment area radius |
| `NUM_USERS` | 100 | Default IoT device count |
| `NUM_UAVS` | 5 | Default UAV count |
| `NUM_JAMMERS` | 1 | Number of jammers |

### Diffusion Model

| Parameter | Default | Description |
|---|---|---|
| `DIFFUSION_DENOISING_STEPS` | 3 | Reverse VP-SDE denoising steps K |
| `DIFFUSION_BETA_A` | 0.1 | Noise schedule lower bound |
| `DIFFUSION_BETA_B` | 20.0 | Noise schedule upper bound |
| `DIFFUSION_HIDDEN_DIM` | 256 | Hidden dimension of denoising network |

### MAPPO Training

| Parameter | Default | Description |
|---|---|---|
| `MAPPO_LR_ACTOR` | 3e-4 | Actor learning rate |
| `MAPPO_LR_CRITIC` | 1e-4 | Critic learning rate |
| `MAPPO_GAMMA` | 0.99 | Discount factor |
| `MAPPO_GAE_LAMBDA` | 0.95 | GAE λ for advantage estimation |
| `MAPPO_CLIP_EPS` | 0.2 | PPO clipping epsilon |
| `MAPPO_ENTROPY_COEF` | 0.01 | Entropy bonus coefficient |
| `MAPPO_PPO_EPOCHS` | 4 | PPO update epochs per rollout |
| `MAPPO_ROLLOUT_LENGTH` | 500 | Steps per rollout (one full episode) |
| `TOTAL_TIMESTEPS` | 5 000 000 | Total training environment steps |

### Caching & Two-Timescale

| Parameter | Default | Description |
|---|---|---|
| `CACHE_UPDATE_INTERVAL` | 25 | Macro-interval length K_C (slots) |
| `NUM_SERVICE_PROGRAMS` | 20 | Total cacheable service programs |
| `ZIPF_EXPONENT` | 1.0 | Zipf popularity distribution exponent |
| `MAB_EXPLORATION_COEFF` | 1.0 | UCB exploration coefficient |

### Objective Weights

| Parameter | Default | Description |
|---|---|---|
| `ALPHA_LATENCY` | 0.6 | Latency weight in cost function |
| `BETA_ENERGY` | 0.4 | Energy weight in cost function |

### Ablation Flags

| Flag | Default | Effect when `False` |
|---|---|---|
| `USE_DIFFUSION` | `True` | Switches to `GaussianOnlyActor` |
| `USE_COOP_CACHE` | `True` | Disables A2A tier-2 cache fetch |
| `USE_LEARNED_TRAJ` | `True` | Replaces MAPPO trajectory with fixed circular orbit |

---

## Environment Details

### IoT Devices
- **Mobility:** Gauss-Markov AR-1 model (speed memory ρ_v = 0.9, heading memory ρ_θ = 0.75, mean speed 1 m/slot)
- **Transmit power:** Variable per device, 15 – 23 dBm
- **Task model:** Data size 0.5 – 5 MB, CPU cycles 0.1 – 1 GHz, deadline 0.2 – 2 s, generation probability 0.7

### UAV Fleet
- **Heterogeneous profiles:** 8 types covering CPU 1.2 – 3.2 GHz, cache 60 – 160 MB, bandwidth 8 – 16 MHz, altitude 85 – 115 m
- **Propulsion model:** Rotary-wing model with blade profile, induced power, and parasite drag components
- **Wind effect:** Ornstein-Uhlenbeck process adds drag to propulsion energy and degrades LoS probability via antenna jitter

### Channel Model
- **A2G path loss:** Probabilistic LoS/NLoS with environment constants (A = 9.62, B = 0.158), Rician K = 10
- **Jammer:** Flat interference within 120 m radius, penalty zone within 180 m
- **NOMA:** Optimal power split α computed by bisection; near user decodes via SIC

### Domain Randomisation (applied every 100 000 steps)
- UAV count, IoT count, jammer count, jammer power, area radius, IoT TX power, noise PSD, wind speed, jammer interference radius all randomised within configured bounds

---

## Output Structure

```
results/
├── checkpoints/
│   └── <ablation>/
│       ├── best_model.pt        # Highest reward checkpoint
│       ├── final_model.pt       # End-of-training checkpoint
│       └── model_step_*.pt      # Periodic interval checkpoints
├── logs/
│   └── <ablation>/
│       └── training_log.jsonl   # Per-step metrics (reward, latency, energy, cache hit)
├── plots/                       # Trajectory plots generated during training
├── videos/                      # 3D flight animation videos
└── test_results/
    └── <checkpoint>_<timestamp>/
        ├── metrics.json         # Summary statistics (mean ± std)
        ├── episodes.jsonl       # Per-episode results
        ├── logs/
        ├── trajectories/
        └── plots/
```

---

## License

This code is released for academic and research use only.
