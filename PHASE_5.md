# Phase 5 — Service Caching and Replication

## Objective

Add per-UAV service caching and replication to the service chain MEC simulator. Tasks dynamically select the best UAV hosting the required service (minimizing A2A communication distance), reducing inter-UAV communication latency while respecting finite cache capacities.

## Implemented functionality

- **Per-UAV Service Cache**: Each UAV maintains a finite `cache_capacity` and a `service_cache` set.
- **Service Storage Requirements**: Configurable `service_sizes` dictionary defining storage requirements per service.
- **Initial Service Placement**: Deterministic initial placement (UAV0 $\to$ A, UAV1 $\to$ B, UAV2 $\to$ C).
- **Service Replication**: Added `replicate_service(uav_id, service_name)` and `evict_service(uav_id, service_name)` respecting storage limits.
- **Configurable Initial Replicas**: `initial_replicas` parameter on `MultiUAVMECEnv`.
- **Dynamic UAV Stage Selection**: `select_best_uav_for_stage(task, stage_idx)` dynamically chooses the closest candidate UAV hosting the required service, reducing A2A distance.
- **Cache Hit / Miss Tracking**: Explicitly tracks local hits, cooperative hits, and cache misses.
- **Replica Count Metrics**: Added `replica_count` and `replica_counts` to `info`.

## Modified files

- `src/env.py`: Added cache capacity, service storage models, replication methods, dynamic stage execution mapping, and caching metrics in `info`.
- `config.py`: Updated `SIMULATOR_PHASE = 5`.

## New files

- `tests/test_phase5_caching.py`
- `PHASE_5.md`

## Tests

File: `tests/test_phase5_caching.py`

1. `test_cache_creation_and_capacity`
2. `test_initial_service_placement`
3. `test_replication_creates_valid_copies`
4. `test_storage_capacity_respected`
5. `test_cached_service_executes_on_replica`
6. `test_uncached_service_cannot_execute`
7. `test_a2a_latency_uses_actual_uavs`
8. `test_replica_reduces_a2a_latency`
9. `test_cache_hit_miss_metrics`
10. `test_replica_count_metrics`
11. `test_task_chain_ordering_preserved`
12. `test_previous_phases_regression`
13. `test_deterministic_smoke_phase5`

Command: `.venv/bin/python tests/test_phase5_caching.py`

## Next phase

Phase 6 — Two-Timescale Control.

PHASE 5 STATUS: PASS
