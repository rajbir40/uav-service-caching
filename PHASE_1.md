# Phase 1 — Basic UAV-MEC

## Objective

Replace the live environment with a minimal, independently runnable multi-UAV MEC simulator: stationary IoT users, mobile UAVs, dynamic task arrivals, A2G upload, FIFO compute, energy/battery, and a CTDE-compatible observation/action interface.

## Implemented functionality

- Stationary IoT users (positions drawn at reset and frozen).
- Mobile rotary-wing UAVs with 3D trajectory commands.
- Bernoulli per-user task arrivals each slot.
- Nearest-UAV association (`src/association.py`).
- A2G uplink into a per-UAV upload buffer; one active upload per UAV.
- FIFO CPU processing; one computing task per UAV.
- UAV flight energy (rotary-wing propulsion), communication energy (RX while uploading), computation energy (DVFS).
- Battery drain, clipped at zero.
- Basic end-to-end latency `T_total = T_upload + T_queue + T_compute` with `T_A2A = 0`.
- CTDE interface: `reset()` → `(obs, state)`, `step(actions)` → `(obs, state, rewards, done, info)`.

## Modified files

- `src/env.py` — live `MultiUAVMECEnv` (Phase 1 core; later extended in Phase 2).
- `src/channel_model.py` — added single-user `a2g_uplink_rate` (no NOMA/jammer in the live env).
- `config.py` — Phase-1 constants (`PHASE1_*`, `SIMULATOR_PHASE`).
- `src/metrics.py` — propulsion and computation energy helpers used by the env.

## New files

- `src/latency.py` — modular latency helpers.
- `src/association.py` — nearest-UAV assignment.
- `tests/test_phase1_env.py` — Phase 1 regression tests.

## Important equations / assumptions

- Users are ground-fixed. UAVs move inside a disk of radius `0.88 * AREA_RADIUS`, altitude in `[UAV_ALT_MIN, UAV_ALT_MAX]`.
- A2G rate: Shannon capacity `R = B log2(1 + P g / (N0 B))` from probabilistic LoS/NLoS gain. Live env forces `wind_speed = 0` and `interference_power = 0`.
- `T_upload = D / R_A2G` when that is ≤ one slot; otherwise wall-clock upload elapsed time.
- `T_compute = C / f` when that is ≤ one slot; otherwise wall-clock compute elapsed time.
- Slot duration: `SLOT_DURATION = TOTAL_TIME / NUM_TIME_SLOTS` (1 s with defaults 500/500).
- Reward: `-(α L̃ + β Ẽ)` with `α = ALPHA_LATENCY`, `β = BETA_ENERGY`, latency/energy normalised by `PHASE1_LAT_NORM` / `PHASE1_ENERGY_NORM`. Shared scalar copied to all active agents.
- Battery: `PHASE1_UAV_BATTERY_J = 5.0e5` J.
- Scheduling is heuristic (nearest + FIFO). Action pads `a[3], a[4]` are unused.

Intentionally **not** in Phase 1: user mobility, NOMA/SIC, jammers, wind dynamics, A2A delay, service chains, caching/replication, two-timescale control, MAPPO/diffusion in the env loop.

## Observation / action changes

Per-agent action (length 5, values in `[-1, 1]`):

- `a[0], a[1]` — x/y velocity commands (× `max_displacement` m/slot)
- `a[2]` — altitude change (× `UAV_ALT_DELTA_MAX`)
- `a[3], a[4]` — unused MAPPO-width padding

Local observation dim **91** (`MAX_UAVS = 8`, `PHASE1_OBS_USERS = 10`):

- own: x, y, alt, vx, vy, battery, cpu_util (7)
- active mask (1)
- other UAV relative xy (2 × 7) and other active masks (7)
- nearest users × (rel_x, rel_y, pending, size, cycles, deadline) (10 × 6)
- queue length (1), time fraction (1)

Global state dim **622** (padded to `MAX_UAVS` and `MAX_NUM_USERS`). Observation tensors are always shape `(MAX_UAVS, 91)`.

## Tests

File: `tests/test_phase1_env.py`

1. `test_obs_action_shapes`
2. `test_user_positions_frozen`
3. `test_uav_moves_with_nonzero_action`
4. `test_a2g_distance_changes_with_uav`
5. `test_rate_decreases_with_distance`
6. `test_compute_latency_scales_with_cycles`
7. `test_energy_and_battery`
8. `test_total_latency_identity`
9. `test_no_noma_sic_jammer_wind_a2a`
10. `test_smoke_print`

Command: `.venv/bin/python tests/test_phase1_env.py`

## Exact test result

```
All 10 Phase-1 tests passed.
```

(Reconfirmed during Phase 2 documentation cleanup.)

## Known limitations

- Queue wait in Phase 1 was incomplete relative to wall-clock spanning; Phase 2 tightens timestamps and identity.
- Depleted-battery UAVs could still be assigned tasks in Phase 1; Phase 2 adds battery-aware association.
- Legacy DMJO code remains in the repo unused by `MultiUAVMECEnv`.
- `README.md` still describes the original DMJO paper stack.
- `ChannelModel` still contains wind-degraded LoS formulas; the live env zeros wind.

## Next phase

Phase 2 — robust task/queue dynamics: explicit timestamps, queue latency, deadline tracking, percentiles, battery-aware association.

PHASE 1 STATUS: PASS
