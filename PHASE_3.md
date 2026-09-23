# Phase 3 — Ordered Service Chains

## Objective

Add ordered service-chain execution ($A \to B \to C$) to the live multi-UAV MEC environment while preserving all Phase 1 and Phase 2 dynamics, timestamps, queues, and latency identities. A2A communication delay remains strictly zero in this phase ($T_{\text{A2A}} = 0$).

## Implemented functionality

- Ordered service chains: default chain $A \to B \to C$ with fixed deterministic UAV placement:
  - $\text{UAV}_0 \to A$
  - $\text{UAV}_1 \to B$
  - $\text{UAV}_2 \to C$
- Modular `ServiceStage` entity tracking stage name, assigned UAV ID, required/remaining CPU cycles, start/finish timestamps, compute latency, queue latency, energy consumption, and stage lifecycle status.
- `Task` stores the chain sequence, stage objects, and `current_stage_idx`.
- Sequential stage-by-stage computation: Stage $A$ executes on $\text{UAV}_0$, then transitions to $\text{UAV}_1$ for Stage $B$, then to $\text{UAV}_2$ for Stage $C$.
- Per-UAV FIFO queues: each UAV manages its own upload buffer and compute queue (`task_queue`).
- Multi-slot task and stage persistence: tasks cleanly span across slot boundaries between stages without losing state or progress. Forwarded stages are staged at slot boundaries and picked up in subsequent slots.
- Precise timestamps per stage: `enqueue_time`, `compute_start_time`, `compute_finish_time`.
- Stage queue latency: $T_{\text{queue}}(\text{stage } 0) = \text{upload\_buffer\_wait} + \text{cpu\_queue\_wait}$, and $T_{\text{queue}}(\text{stage } s) = \text{cpu\_queue\_wait}$ for subsequent stages.
- Stage compute latency: closed form $C_s / f_u$ if $\le \text{SLOT\_DURATION}$, else wall-clock elapsed compute time.
- Task completion only occurs after the final stage ($C$) finishes.
- Computation energy accounted for on every stage and charged to the respective executing UAV.
- Chain metrics in `info`: `chain_avg_latency`, `chain_p50_latency`, `chain_p95_latency`, `chain_p99_latency`, `chains_completed`, `episode_chains_completed`.
- Strict latency identity:
  $$T_{\text{total}} = T_{\text{upload}} + \sum_{s} T_{\text{queue}}(s) + \sum_{s} T_{\text{compute}}(s) + T_{\text{A2A}}$$
  with $T_{\text{A2A}} = 0.0$.

## Modified files

- `config.py` — added `DEFAULT_SERVICE_CHAIN`, `DEFAULT_SERVICE_PLACEMENT`, updated `SIMULATOR_PHASE = 3`.
- `src/env.py` — added `ServiceStage` class, updated `Task` with chain/stages, stage routing in `_process_uploads`, stage-by-stage compute and queue handling in `_process_compute`, and chain metrics in `_pack_reward_info`.

## New files

- `tests/test_phase3_service_chain.py` — 12 Phase-3 service chain validation tests.
- `PHASE_3.md` — this documentation.

## Important equations / assumptions

- Default chain: $A \to B \to C$ with fixed placement:
  $$\text{placement}(A) = 0, \quad \text{placement}(B) = 1, \quad \text{placement}(C) = 2$$
- Per-stage cycle allocation:
  $$C_{\text{stage}} = \frac{C_{\text{task}}}{|\text{chain}|}$$
- Latency identity:
  $$T_{\text{total}} = T_{\text{upload}} + \sum_{s \in \text{stages}} T_{\text{queue}}(s) + \sum_{s \in \text{stages}} T_{\text{compute}}(s) + T_{\text{A2A}}$$
  where $T_{\text{A2A}} = 0.0$.
- Queue wait breakdown:
  - Stage 0: $T_{\text{queue}}(0) = \max(0, t_{\text{upload\_start}} - t_{\text{arrival}}) + \max(0, t_{\text{compute\_start}, 0} - t_{\text{enqueue}, 0})$
  - Stage $s > 0$: $T_{\text{queue}}(s) = \max(0, t_{\text{compute\_start}, s} - t_{\text{enqueue}, s})$
- Computation energy:
  $$E_{\text{comp}, s} = \kappa \cdot f_u^2 \cdot C_s$$
  charged to $\text{UAV}_u$ executing stage $s$.
- Intentionally **not** in Phase 3: distance-based A2A communication rate, actual A2A latency, caching, replication, cache optimization, MAPPO, diffusion, NOMA/SIC, jammers, wind dynamics, user mobility, two-timescale control.

## Observation / action changes

None. Local observations remain shape `(MAX_UAVS, 91)` and global state remains dimension `622`. Action dimensions remain `5`.

## Tests

File: `tests/test_phase3_service_chain.py`

1. `test_chain_creation` — Task initializes chain A->B->C with correct stages and placement
2. `test_chain_ordering` — Stage execution order: A finishes before B starts, B before C
3. `test_each_stage_executes_once` — Every stage executes exactly once per completed task
4. `test_correct_uav_placement` — UAV0 hosts A, UAV1 hosts B, UAV2 hosts C
5. `test_stage_persistence_across_slots` — Tasks persist in queues across slots between stages
6. `test_no_premature_completion` — No task is marked completed until stage C finishes
7. `test_correct_stage_latency` — Individual stage compute and queue latencies are accurate
8. `test_total_latency_identity_phase3` — Strict identity holds for all completed tasks
9. `test_computation_energy_all_stages` — Computation energy accounted for on every stage and UAV
10. `test_chain_metrics` — info dict exposes chain avg, p50, p95, p99, and completion counts
11. `test_chain_percentiles` — Percentile ordering $p_{99} \ge p_{95} \ge p_{50} \ge 0$
12. `test_deterministic_smoke_phase3` — 100-slot deterministic run: no NaN/Inf, all invariants hold

Commands:

```powershell
python tests/test_phase1_env.py
python tests/test_phase2_env.py
python tests/test_phase3_service_chain.py
```

## Exact test result

```
All 10 Phase-1 tests passed.
All 16 Phase-2 tests passed.
All 12 Phase-3 tests passed.
```

## Known limitations

- Service placement is fixed ($\text{UAV}_0 \to A$, $\text{UAV}_1 \to B$, $\text{UAV}_2 \to C$).
- A2A transfer delay between UAVs is zero ($T_{\text{A2A}} = 0$).
- Services are not yet replicated across multiple UAVs.
- Caching decisions are not yet dynamic.

## Next phase

Phase 4 — Explicit A2A: Add actual UAV-to-UAV communication delay ($T_{\text{A2A}} = D_{\text{intermediate}} / R_{\text{A2A}}$) where inter-UAV communication rates depend dynamically on 3D Euclidean distance and trajectory.

PHASE 3 STATUS: PASS

