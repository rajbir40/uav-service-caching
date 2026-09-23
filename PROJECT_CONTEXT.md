# Multi-UAV MEC Research Simulator — Project Context

## Research Direction

We are developing a research simulator for **service-chain replication and tail-latency-aware multi-UAV Mobile Edge Computing (MEC)**.

Core idea:

- Fixed, stationary IoT users.
- Dynamic task arrivals create network dynamicity.
- Multiple mobile UAVs provide MEC services.
- Tasks consist of ordered service chains, e.g. A → B → C.
- Different chain stages may execute on different UAVs.
- UAV-to-UAV (A2A) communication is required when consecutive stages are placed on different UAVs.
- UAV relative positions affect A2A communication rate.
- Therefore UAV trajectory affects inter-UAV communication latency, which affects service-chain latency.
- Service caching/replication will later allow multiple UAVs to host the same service.
- Eventually optimize UAV trajectory, service placement/replication, task execution, computation resources, and tail latency using multi-agent RL.

The term "kinematic tethering" is our conceptual description of the coupling between UAV movement and service-chain communication latency. It is NOT claimed to be an established literature term.

## Important Scope Decisions

Keep the core simulator simple.

### Included

- Stationary IoT users
- Dynamic task arrivals
- Mobile UAVs
- A2G user → UAV communication
- UAV computation
- Per-UAV queues
- Service chains
- A2A UAV → UAV communication
- Service caching
- Service replication
- UAV trajectory
- UAV energy
- Task/service latency
- Tail latency metrics
- Eventually MAPPO / multi-agent RL
- Eventually two-timescale control

### NOT part of the current core

- User mobility
- Dynamic user ON/OFF
- WPT
- RIS
- UGV
- HAP
- Fixed-wing UAVs
- D2D
- Cloud hierarchy
- NOMA/SIC
- Jammers
- Wind
- Diffusion actor
- Complex legacy peer/BS fetching

User population stays fixed and stationary.

Dynamicity comes from:

- task arrivals
- queues
- service demand
- cache state
- UAV trajectory
- UAV battery
- computation state

## Development Strategy

We are NOT rewriting the entire project at once.

The original DMJO repository is being converted incrementally.

Each phase must:

1. Be independently runnable.
2. Preserve previous phases.
3. Have dedicated tests.
4. Pass all previous tests.
5. Stop before implementing the next phase.

Never jump ahead.

The live simulator is `src/env.py` (`MultiUAVMECEnv`). Original DMJO modules (`src/legacy_env.py`, `src/caching.py`, `src/diffusion.py`, `src/mappo_agent.py`, NOMA in `src/channel_model.py`, `train.py`, `README.md`) remain in the tree for incremental reuse. They are **not** the Phase 1/2 core.

## Phase Roadmap

### Phase 0 — Understand/Frozen Original

Understand the existing DMJO repository and identify reusable components.

### Phase 1 — Basic UAV-MEC

Implemented. See `PHASE_1.md`.

### Phase 2 — Robust Task and Queue Dynamics

Implemented. See `PHASE_2.md`.

### Phase 3 — Service Chains

Implemented. See `PHASE_3.md`.

Add ordered service-chain execution but NO A2A delay yet.

Example:

A → B → C

Fixed deterministic placement:

UAV0 → A
UAV1 → B
UAV2 → C

A2A latency remains zero in this phase.

### Phase 4 — Explicit A2A

Add actual UAV-to-UAV communication.

A2A delay:

T_A2A = D_intermediate / R_A2A

Rate depends on UAV-UAV distance.

### Phase 5 — Caching and Replication

Add service caching and replication.

Multiple UAVs can host the same service.

### Phase 6 — Two-Timescale Control

Introduce macro service-placement decisions and micro trajectory/execution decisions.

### Phase 7 — MAPPO

Integrate multi-agent reinforcement learning.

### Phase 8 — Baselines and Experiments

Implement baselines and research experiments.

### Optional Phase 9 — Advanced Methods

Diffusion or other improvements only if justified later.

## Current Status

Phase 1: PASS
Phase 2: PASS
Phase 3: PASS
Phase 4: NEXT

## Development Rule

Do not introduce mechanisms from future phases early.

Especially:

Phase 3 must NOT contain:

- A2A latency
- service replication
- cache optimization
- MAPPO
- diffusion
- NOMA/SIC
- jammers
- wind
- user mobility
- two-timescale control

After completing a phase, create/update its corresponding `PHASE_X.md` file with:

- objective
- implemented functionality
- modified files
- new files
- important equations/assumptions
- observation/action changes
- tests
- exact test result
- known limitations
- next phase

Always finish with:

PHASE X STATUS: PASS

or

PHASE X STATUS: FAIL
