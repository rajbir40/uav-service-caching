# Phase 0 — Repository Baseline and Stabilization

## 1. Current Architecture
The repository is an incremental research simulator for Multi-UAV Mobile Edge Computing (MEC). It currently features:
- **Stationary IoT Users**: Fixed ground positions across time slots.
- **Mobile UAVs**: Continuous 2D trajectory control.
- **Communication Channels**: Air-to-Ground (A2G) and Air-to-Air (A2A) path-loss and LoS/NLoS models.
- **Task Execution & Queues**: Per-UAV task queues and CPU capacity allocation.
- **Caching & Replication**: Per-UAV cache storage capacities and service placement.
- **Multi-Agent RL**: MAPPO with Centralized Training, Decentralized Execution (CTDE), alongside a Diffusion-based hybrid actor and baseline agents (MADDPG, MATD3, Greedy, Local).

## 2. Important Modules
- `src/env.py`: Core OpenAI Gym-compatible environment (`MultiUAVMECEnv`), UAV, IoTDevice, Task, and ServiceStage models.
- `src/config.py`: Central configuration constants (area, UAV profiles, communication, reward weights, ablation flags, diffusion settings).
- `src/channel_model.py`: Channel models (`ChannelModel`, `NOMACommunication`).
- `src/mappo_agent.py`: MAPPO agent implementation with CentralisedCritic.
- `src/diffusion.py`: Diffusion-Gaussian hybrid actor (`DiffusionActor`, `GaussianOnlyActor`, `VPNoiseSchedule`).
- `src/caching.py`: Legacy cooperative caching managers (`FrequencyWeightCaching`, `CooperativeCacheManager`).
- `src/baselines/`: Baseline agents (MADDPG, MATD3, Greedy, Local, MAPPO baseline).
- `tests/`: Phase-specific test suites (`test_phase1_env.py` to `test_phase5_caching.py`).

## 3. Current State, Action, and Reward
- **State (`get_local_obs`, `get_global_state`)**: Local UAV observations include position, velocity, energy, CPU utilization, cache states, and packed nearby user features. Global state includes full fleet and user positions.
- **Action (`step`)**: Continuous trajectory velocity actions $(\Delta x, \Delta y, \dots)$ per UAV.
- **Reward (`_pack_reward_info`)**: Weighted combination of mean task latency and fleet energy consumption: $R = -(\alpha L + \beta E)$.

## 4. Caching, Offloading, Trajectory, and Energy Behavior
- **Caching**: UAVs maintain a cache capacity and service cache. Phase 5 introduced replication and `select_best_uav_for_stage` to map tasks to available replicas.
- **Offloading**: Tasks are generated at stationary IoT devices, uploaded via A2G links, and processed in UAV queues.
- **Trajectory**: UAVs move continuously based on velocity actions, bounded by maximum displacement and area limits.
- **Energy**: Rotary-wing propulsion power, communication TX/RX power, and computation power drawn from per-UAV batteries.

## 5. RL and Baselines
- **MAPPO**: On-policy multi-agent reinforcement learning using CTDE with GAE and PPO clip updates.
- **Diffusion Actor**: Hybrid Gaussian-diffusion policy head (`src/diffusion.py`).
- **Baselines**: MADDPG, MATD3, Greedy, and Local execution agents in `src/baselines/`.

## 6. Tests and Results
- **Phase 1 Tests**: 10/10 PASS
- **Phase 2 Tests**: 16/16 PASS
- **Phase 3 Tests**: 12/12 PASS
- **Phase 4 Tests**: 10/10 PASS
- **Phase 5 Tests**: 13/13 PASS
- **Total Existing Tests**: 61/61 PASS. Environment successfully initializes, resets, steps, and terminates without NaN/Inf errors.

## 7. Legacy Components and Their Status

| Component | Location | Status | Reason / Alignment with `research_paper.md` |
| :--- | :--- | :--- | :--- |
| Service-Chain / Multi-Stage Execution | `src/env.py` (`ServiceStage`, `stages_to_forward`, `select_best_uav_for_stage`) | `REMOVE LATER` | `research_paper.md` specifies atomic single-service requests ($s \in \mathcal{S}$), not multi-stage DAG service chains. |
| User Mobility | `config.py`, `src/env.py` (Gauss-Markov) | `REMOVE LATER` | `research_paper.md` specifies stationary IoT devices with active/inactive hotspot demand dynamics, not user mobility. |
| WPT / Energy Harvesting | `config.py`, `src/env.py` | `REMOVE LATER` | `research_paper.md` explicitly removes WPT and device-side energy harvesting from the model. |
| Legacy NOMA Functionality | `src/channel_model.py` (`NOMACommunication`) | `REMOVE LATER` | NOMA/SIC allocation is outside the scope of `research_paper.md`. |
| Diffusion Actor | `src/diffusion.py`, `config.py` (`USE_DIFFUSION`) | `REMOVE LATER` | `research_paper.md` specifies attention-based MAPPO, not diffusion-enhanced actor heads. |
| Legacy Caching Mechanisms | `src/caching.py` (`FrequencyWeightCaching`, etc.) | `REMOVE LATER` | Replaced by coordinated, persistent atomic-service replication model. |
| Mean-Latency Reward | `src/env.py` (`_pack_reward_info`), `config.py` (`ALPHA_LATENCY`, `BETA_ENERGY`) | `REMOVE LATER` | `research_paper.md` specifies $\text{CVaR}_{0.95}$ tail-latency objective + switch cost + energy penalty. |
| Chain-Based Metrics | `src/env.py` (`chain_*_latency`) | `REMOVE LATER` | Replaced by device-level atomic-task CVaR/P95/P99 latency metrics. |
| Phase Flags & Stale Configs | `config.py` (`SIMULATOR_PHASE`, etc.) | `REMOVE LATER` | Incremental phase development flags are no longer needed once transitioning to the final research paper specification. |
| MAPPO Core / CTDE / Buffers | `src/mappo_agent.py`, `src/buffer.py` | `KEEP` | Core reinforcement learning infrastructure is required for Attention-MAPPO. |
| UAV Trajectory & Energy | `src/env.py`, `config.py` | `KEEP` | Continuous movement, collision constraints, and propulsion/comm/comp energy are core components. |
| A2G & A2A Channels | `src/channel_model.py` (`ChannelModel`) | `KEEP` | Communication path loss and rate calculations are required. |

## 8. Known Issues
- The repository currently implements multi-stage service chains (Phases 3–5) and diffusion actors, which must be removed in subsequent phases per `research_paper.md`.
- Reward function currently optimizes mean latency rather than CVaR tail latency.

## 9. Phase 1 Prerequisites (for Future Transition)
- Clean up obsolete service-chain and multi-stage abstractions.
- Implement atomic single-service task model.
- Replace mean-latency reward with $\text{CVaR}_{0.95}$ tail objective.
- Integrate attention mechanism in observations and actor-critic networks.
- Implement action masking for cache capacity, migration, and task assignment.

PHASE 0 STATUS: PASS
