# Phase 2 — Robust Task and Queue Dynamics

## Objective

Keep the Phase 1 UAV-MEC loop and make task lifecycle, queues, latency identity, deadlines, and battery constraints explicit and testable. Do not add service chains or A2A delay.

## Implemented functionality

- Explicit task lifecycle: generated → assigned → uploading → queued → computing → completed.
- Persistent FIFO upload and CPU queues; unfinished work spans multiple slots.
- Wall-clock timestamps: `upload_start_time`, `upload_finish_time`, `enqueue_time`, `compute_start_time`, `compute_finish_time`.
- Queue latency = upload-buffer wait + CPU-queue wait.
- Strict identity: `t_total == t_upload + t_queue + t_compute + t_a2a` with `t_a2a == 0`.
- Unique monotonic `task_id`; `remaining_bits` / `remaining_cycles` must be ~0 on completion.
- Per-task `deadline_missed` iff `t_total > deadline`.
- Episode counters: generated, completed, deadline misses; pending backlog in `info`.
- Episode latency percentiles: `p50_latency`, `p95_latency`, `p99_latency`.
- Battery-aware association: UAVs with `battery_energy <= 0` do not move and do not receive new upload-buffer tasks. If all UAVs are depleted, the arrival is dropped.
- `info` aliases for energy and task counts (Phase 1 keys preserved).

## Modified files

- `src/env.py` — Phase-2 `Task` fields, step internals, episode accumulators, `info` dict. Public `reset`/`step` shapes unchanged.
- `config.py` — `SIMULATOR_PHASE = 2` (set during documentation cleanup; env behaviour already Phase 2).

## New files

- `tests/test_phase2_env.py` — 16 Phase-2 tests.

## Important equations / assumptions

Let `now = t * SLOT_DURATION` at the end of the current slot.

Upload-buffer wait:

`max(0, upload_start_time - arrival_time)`

CPU-queue wait:

`max(0, compute_start_time - upload_finish_time)`

`T_queue` is the sum of those two waits (not only `compute_start - enqueue_time`).

`T_upload`: instantaneous `D/R` if that is ≤ `SLOT_DURATION`, else `upload_elapsed`.

`T_compute`: `C/f` if that is ≤ `SLOT_DURATION`, else `compute_elapsed`.

`T_A2A = 0` (`calculate_a2a_latency()` placeholder for Phase 4).

`T_total = T_upload + T_queue + T_compute + T_A2A`.

Completion only when `remaining_bits ≈ 0` and `remaining_cycles ≈ 0`.

Association: nearest UAV among **active UAVs with battery > 0**. Users remain stationary. No NOMA, jammers, wind evolution, chains, or caching in the live env.

Reward is unchanged from Phase 1 (mean completed-task latency this slot + fleet energy). Percentiles are reported in `info`, not yet in the reward.

## Observation / action changes

None. Tests require:

- `obs.shape == (MAX_UAVS, 91)` with `MAX_UAVS = 8`
- `env.agent_action_dim == 5`
- global state still padded to `MAX_UAVS` / `MAX_NUM_USERS`

## Tests

File: `tests/test_phase2_env.py`

1. `test_obs_action_shapes`
2. `test_users_stationary_p2`
3. `test_unique_task_ids`
4. `test_tasks_persist_across_steps`
5. `test_unfinished_tasks_stay_queued`
6. `test_queue_grows_high_load`
7. `test_queue_drains_low_load`
8. `test_no_premature_completion`
9. `test_no_negative_queue_latency`
10. `test_latency_identity`
11. `test_completion_leq_generated`
12. `test_deadline_tracking`
13. `test_percentile_ordering`
14. `test_battery_never_negative`
15. `test_depleted_uav_no_new_tasks`
16. `test_100_slot_deterministic`

Phase 1 suite must still pass.

Commands:

```
.venv/bin/python tests/test_phase1_env.py
.venv/bin/python tests/test_phase2_env.py
```

## Exact test result

```
All 10 Phase-1 tests passed.
All 16 Phase-2 tests passed.
```

(Reconfirmed during documentation cleanup.)

## Known limitations

- Tasks are still a single compute blob (no service chain stages).
- A2A latency is hardcoded to zero.
- Dropped arrivals when all UAVs are depleted are not counted as generated.
- `SIMULATOR_PHASE` in `config.py` lagged at `1` until this documentation cleanup; env was already Phase 2.
- `PROJECT_CONTEXT.md` / `PHASE_*.md` did not exist until this cleanup.
- `README.md` still describes original DMJO (NOMA, jammers, wind, diffusion).
- Legacy modules (`legacy_env`, caching, diffusion, MAPPO trainer) are unused by the live env.
- `src/__init__.py` still re-exports those modules.
- Wind LoS terms remain in `ChannelModel` but `env.channel.wind_speed = 0.0`.

## Next phase

Phase 3 — ordered service-chain execution with **fixed** placement (UAV0→A, UAV1→B, UAV2→C) and **zero** A2A delay. Do not add replication, cache optimization, MAPPO, or distance-based A2A.

PHASE 2 STATUS: PASS
