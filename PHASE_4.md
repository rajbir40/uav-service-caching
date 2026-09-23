# Phase 4 — Explicit A2A Communication

## Objective

Add explicit UAV-to-UAV (A2A) communication delay to the service chain ($A \to B \to C$) MEC simulator. A2A communication rate is dynamically computed based on 3D Euclidean distance.

## Implemented functionality

- Added `a2a_rate()` to `ChannelModel`, implementing the Shannon capacity formula for A2A communication ($R_{A2A} = B_{A2A} \log_2(1 + P_{tx} g_{A2A} / (N_0 B_{A2A}))$), where $g_{A2A} = \eta_{LoS} d^{-l}$ (pure LoS path loss).
- Updated `ServiceStage` to track `t_a2a` (explicit A2A latency), `a2a_rate`, and `a2a_distance`.
- Updated `Task` to include `t_a2a` as a property (sum of stage transition A2A latencies).
- Updated `MultiUAVMECEnv._process_compute` to add A2A latency when consecutive stages are on different UAVs: $T_{A2A} = D_{intermediate} / R_{A2A}$.
- A2A communication energy is accounted for separately and added to `e_comm`: TX energy to the transmitter UAV and RX energy to the receiver UAV.
- Added `a2a_latency`, `a2a_rate`, and `a2a_distance` metrics to `info`.

## Modified files

- `src/latency.py`: Implemented `calculate_a2a_latency`.
- `src/channel_model.py`: Added `a2a_rate` to `ChannelModel`.
- `src/env.py`: Updated `Task`, `ServiceStage` classes, `MultiUAVMECEnv` logic for A2A latency and energy, and `info` dict.
- `tests/test_phase3_service_chain.py`: Updated assertions to handle non-zero A2A latency.

## New files

- `tests/test_phase4_a2a.py`
- `PHASE_4.md`

## Important equations

- $R_{A2A} = B_{A2A} \log_2(1 + \frac{P_{tx} \eta_{LoS} d^{-l}}{N_0 B_{A2A}})$
- $T_{A2A} = \frac{D_{intermediate}}{R_{A2A}}$
- $E_{comm, A2A} = P_{TX} T_{A2A} + P_{RX} T_{A2A}$

## Tests

File: `tests/test_phase4_a2a.py`

1. `test_a2a_rate_positive`
2. `test_a2a_rate_decreases_with_distance`
3. `test_a2a_delay_identity_dr`
4. `test_abc_two_a2a_transitions`
5. `test_chain_latency_includes_a2a`
6. `test_farther_uavs_increase_a2a_latency`
7. `test_a2a_latency_finite`

Command: `.venv/bin/python tests/test_phase4_a2a.py`

## Next phase

Phase 5 — Caching and Replication.

PHASE 4 STATUS: PASS
