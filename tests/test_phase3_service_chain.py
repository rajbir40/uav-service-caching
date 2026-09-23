"""
Phase-3 ordered service-chain execution validation.
===================================================

Tests (12):
  1.  test_chain_creation                  — Task initializes chain A->B->C with correct stages
  2.  test_chain_ordering                  — Stage execution order: A finishes before B starts, B before C
  3.  test_each_stage_executes_once        — Every stage executes exactly once per completed task
  4.  test_correct_uav_placement           — UAV0 hosts A, UAV1 hosts B, UAV2 hosts C
  5.  test_stage_persistence_across_slots  — Tasks persist in queues across slots between stages
  6.  test_no_premature_completion         — No task is marked completed until stage C finishes
  7.  test_correct_stage_latency           — Individual stage compute and queue latencies are accurate
  8.  test_total_latency_identity_phase3   — T_total = T_upload + Σ T_queue(stage) + Σ T_compute(stage) + T_A2A (T_A2A = 0)
  9.  test_computation_energy_all_stages   — Computation energy accounted for on every stage and UAV
  10. test_chain_metrics                   — info dict exposes chain avg, p50, p95, p99, and completion counts
  11. test_chain_percentiles               — Ordering: p99 >= p95 >= p50 >= 0
  12. test_deterministic_smoke_phase3      — 100-slot deterministic run: no NaN/Inf, all invariants hold

Run:
    python tests/test_phase3_service_chain.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import MultiUAVMECEnv, Task, ServiceStage  # noqa: E402
from config import (  # noqa: E402
    SLOT_DURATION, NUM_TIME_SLOTS, DEFAULT_SERVICE_CHAIN,
    DEFAULT_SERVICE_PLACEMENT,
)


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------
def _finite(*vals):
    for v in vals:
        if isinstance(v, (list, tuple, np.ndarray)):
            arr = np.asarray(v, dtype=float)
            assert np.isfinite(arr).all(), f"non-finite values: {v}"
        else:
            assert math.isfinite(float(v)), f"non-finite value: {v}"


def _zero_actions(env):
    return np.zeros((env.n_agents, env.agent_action_dim), dtype=np.float32)


# ================================================================
# TESTS
# ================================================================

def test_chain_creation():
    """Verify that Task correctly constructs default and custom service chains."""
    # Default chain: A -> B -> C
    Task.reset_ids()
    t = Task(
        user_id=0,
        arrival_time=0.0,
        data_size_bits=1e6,
        cpu_cycles=3e8,
        deadline=2.0,
    )
    assert t.chain == ["A", "B", "C"], f"Expected ['A', 'B', 'C'], got {t.chain}"
    assert len(t.stages) == 3
    assert [s.name for s in t.stages] == ["A", "B", "C"]
    assert t.current_stage_idx == 0
    assert t.current_stage is not None
    assert t.current_stage.name == "A"
    assert t.status == "generated"

    # Verify fixed placement: UAV0 -> A, UAV1 -> B, UAV2 -> C
    assert t.stages[0].uav_id == 0
    assert t.stages[1].uav_id == 1
    assert t.stages[2].uav_id == 2

    # Cycle division
    total_stage_cycles = sum(s.cpu_cycles for s in t.stages)
    np.testing.assert_allclose(total_stage_cycles, t.cpu_cycles, rtol=1e-9)
    for s in t.stages:
        np.testing.assert_allclose(s.cpu_cycles, 1e8, rtol=1e-9)
        assert s.remaining_cycles == s.cpu_cycles
        assert s.status == "pending"

    # Custom chain test
    custom_t = Task(
        user_id=1,
        arrival_time=1.0,
        data_size_bits=2e6,
        cpu_cycles=4e8,
        deadline=3.0,
        chain=["S1", "S2"],
        placement={"S1": 2, "S2": 0},
    )
    assert custom_t.chain == ["S1", "S2"]
    assert len(custom_t.stages) == 2
    assert custom_t.stages[0].uav_id == 2
    assert custom_t.stages[1].uav_id == 0
    np.testing.assert_allclose(sum(s.cpu_cycles for s in custom_t.stages), 4e8, rtol=1e-9)


def test_chain_ordering():
    """Verify strictly ordered execution: A finishes before B starts, B before C."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=30, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0, "Need completed tasks for ordering test"

    for t in env.completed_tasks:
        s0, s1, s2 = t.stages[0], t.stages[1], t.stages[2]
        # Timestamps must exist
        assert s0.compute_start_time is not None
        assert s0.compute_finish_time is not None
        assert s1.compute_start_time is not None
        assert s1.compute_finish_time is not None
        assert s2.compute_start_time is not None
        assert s2.compute_finish_time is not None

        # Stage A: start <= finish
        assert s0.compute_start_time <= s0.compute_finish_time
        # Stage A finish <= Stage B enqueue
        assert s0.compute_finish_time <= s1.enqueue_time
        # Stage B enqueue <= Stage B start <= Stage B finish
        assert s1.enqueue_time <= s1.compute_start_time <= s1.compute_finish_time
        # Stage B finish <= Stage C enqueue
        assert s1.compute_finish_time <= s2.enqueue_time
        # Stage C enqueue <= Stage C start <= Stage C finish
        assert s2.enqueue_time <= s2.compute_start_time <= s2.compute_finish_time


def test_each_stage_executes_once():
    """Verify that each stage A, B, C executes and completes exactly once."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=31, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0

    for t in env.completed_tasks:
        assert t.current_stage_idx == 3, f"Expected stage idx 3 (all completed), got {t.current_stage_idx}"
        for idx, s in enumerate(t.stages):
            assert s.status == "completed", f"Stage {s.name} status={s.status} != 'completed'"
            assert s.remaining_cycles <= 1e-6, f"Stage {s.name} remaining_cycles={s.remaining_cycles}"
            assert s.t_compute > 0.0, f"Stage {s.name} compute latency={s.t_compute} <= 0"
            assert s.compute_elapsed > 0.0


def test_correct_uav_placement():
    """Verify that Stage A computes on UAV0, Stage B on UAV1, Stage C on UAV2."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=32, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0

    for t in env.completed_tasks:
        assert t.stages[0].uav_id == 0, f"Stage A executed on UAV {t.stages[0].uav_id}, expected 0"
        assert t.stages[1].uav_id == 1, f"Stage B executed on UAV {t.stages[1].uav_id}, expected 1"
        assert t.stages[2].uav_id == 2, f"Stage C executed on UAV {t.stages[2].uav_id}, expected 2"


def test_stage_persistence_across_slots():
    """Verify that tasks persist in queues across multiple slots between stages."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=33, task_gen_prob=0.8)
    env.reset()

    saw_intermediate_queued = False
    for slot in range(30):
        env.step(_zero_actions(env))
        # Inspect queues for tasks in intermediate stages (stage B or stage C)
        for uav in env._active_uavs():
            for t in uav.task_queue:
                if t.current_stage_idx in (1, 2):
                    saw_intermediate_queued = True
                    break
            if saw_intermediate_queued:
                break

    assert saw_intermediate_queued, (
        "Expected to observe intermediate-stage tasks (stage 1 or 2) queued across slots"
    )


def test_no_premature_completion():
    """Verify that no task is marked completed until the final stage (C) finishes."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=34, task_gen_prob=0.7)
    env.reset()

    for _ in range(40):
        env.step(_zero_actions(env))

        # Check all in-flight tasks in upload buffers and task queues
        for uav in env._active_uavs():
            for t in list(uav.upload_buffer) + list(uav.task_queue):
                assert t.status != "completed", f"In-flight task {t.task_id} marked completed prematurely"
                assert t.current_stage_idx < len(t.stages)
            if uav.active_upload is not None:
                assert uav.active_upload.status != "completed"
            if uav.computing is not None:
                assert uav.computing.status != "completed"

    # All completed tasks must have completed all stages
    for t in env.completed_tasks:
        assert t.status == "completed"
        assert t.current_stage_idx == len(t.stages)
        assert t.remaining_cycles <= 1e-6
        assert t.remaining_bits <= 1e-9


def test_correct_stage_latency():
    """Verify that stage compute and queue latencies are correctly calculated."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=35, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0

    for t in env.completed_tasks:
        for idx, s in enumerate(t.stages):
            uav = env.uavs[s.uav_id]
            closed = s.cpu_cycles / max(uav.cpu_freq, 1e-12)
            expected_comp = closed if closed <= SLOT_DURATION else s.compute_elapsed
            np.testing.assert_allclose(
                s.t_compute, expected_comp, rtol=1e-5,
                err_msg=f"Task {t.task_id} stage {s.name} compute latency mismatch"
            )
            assert s.t_queue >= 0.0, f"Negative queue wait on stage {s.name}: {s.t_queue}"
            assert s.compute_finish_time >= s.compute_start_time


def test_total_latency_identity_phase3():
    """Verify T_total = T_upload + Σ T_queue(stage) + Σ T_compute(stage) + T_A2A with T_A2A = 0."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=36, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0

    for t in env.completed_tasks:
        # Sum across stages
        sum_stage_queue = sum(s.t_queue for s in t.stages)
        sum_stage_compute = sum(s.t_compute for s in t.stages)

        # Reconstructed total
        recon = t.t_upload + sum_stage_queue + sum_stage_compute + t.t_a2a

        # 1. Total equals reconstructed sum
        np.testing.assert_allclose(
            t.t_total, recon, rtol=1e-9, atol=1e-9,
            err_msg=(
                f"Phase 3 total latency identity failed for task {t.task_id}: "
                f"t_total={t.t_total:.9f} vs sum={recon:.9f}"
            ),
        )

        # 2. Stage sums match task aggregated fields
        np.testing.assert_allclose(t.t_queue, sum_stage_queue, rtol=1e-9, atol=1e-9)
        np.testing.assert_allclose(t.t_compute, sum_stage_compute, rtol=1e-9, atol=1e-9)

        # 3. T_A2A must remain exactly 0.0 in Phase 3
        assert t.t_a2a == 0.0, f"Expected t_a2a == 0.0, got {t.t_a2a}"


def test_computation_energy_all_stages():
    """Verify computation energy is tracked for every stage on its executing UAV."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=37, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0

    uav_stage_energy = {0: 0.0, 1: 0.0, 2: 0.0}
    for t in env.completed_tasks:
        for s in t.stages:
            assert s.energy > 0.0, f"Task {t.task_id} stage {s.name} energy={s.energy} <= 0"
            uav_stage_energy[s.uav_id] += s.energy

    # All three UAVs (0, 1, 2) must have performed computation
    for uav_id in (0, 1, 2):
        assert uav_stage_energy[uav_id] > 0.0, f"UAV {uav_id} has zero stage computation energy"


def test_chain_metrics():
    """Verify info dictionary contains chain metrics and completion counters."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=38, task_gen_prob=0.7)
    env.reset()

    last_info = None
    for _ in range(50):
        _, _, _, _, last_info = env.step(_zero_actions(env))

    assert last_info is not None
    # Required Phase-3 keys
    assert "chain_avg_latency" in last_info
    assert "chain_p50_latency" in last_info
    assert "chain_p95_latency" in last_info
    assert "chain_p99_latency" in last_info
    assert "chains_completed" in last_info
    assert "episode_chains_completed" in last_info

    # Consistency with task counters
    assert last_info["chains_completed"] == last_info["tasks_completed"]
    assert last_info["episode_chains_completed"] == env._episode_tasks_completed
    if env.completed_tasks:
        expected_avg = float(np.mean([t.t_total for t in env.completed_tasks]))
        np.testing.assert_allclose(last_info["chain_avg_latency"], expected_avg, rtol=1e-5)


def test_chain_percentiles():
    """Verify percentile ordering: p99 >= p95 >= p50 >= 0."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=39, task_gen_prob=0.8)
    env.reset()

    last_info = None
    for _ in range(80):
        _, _, _, _, last_info = env.step(_zero_actions(env))

    assert last_info is not None
    p50 = last_info["chain_p50_latency"]
    p95 = last_info["chain_p95_latency"]
    p99 = last_info["chain_p99_latency"]

    assert p50 >= 0.0, f"Negative chain p50: {p50}"
    assert p95 >= p50, f"chain p95={p95} < p50={p50}"
    assert p99 >= p95, f"chain p99={p99} < p95={p95}"


def test_deterministic_smoke_phase3():
    """100-slot deterministic simulation: all invariants and chain properties hold."""
    env = MultiUAVMECEnv(num_users=40, num_uavs=5, seed=42, task_gen_prob=0.7)
    obs, state = env.reset()
    _finite(obs, state)

    prev_completed = 0
    prev_generated = 0

    for slot in range(100):
        obs, state, rewards, done, info = env.step(_zero_actions(env))

        _finite(obs, state, rewards)
        _finite(
            info["total_latency"],   info["energy_consumed"],
            info["upload_latency"],  info["queue_latency"],
            info["compute_latency"], info["chain_avg_latency"],
            info["chain_p50_latency"], info["chain_p95_latency"],
            info["chain_p99_latency"],
        )

        # Battery non-negative
        for b in info["battery"]:
            assert b >= 0.0, f"Slot {slot+1}: negative battery {b}"

        # Monotonicity
        assert env._episode_tasks_generated >= prev_generated
        assert env._episode_tasks_completed >= prev_completed
        prev_generated = env._episode_tasks_generated
        prev_completed = env._episode_tasks_completed

        # A2A remains exactly 0
        assert info["a2a_latency"] == 0.0

    # Invariant: completed <= generated
    assert env._episode_tasks_completed <= env._episode_tasks_generated
    assert len(env.completed_tasks) > 0

    # Latency identity holds for every completed task
    for t in env.completed_tasks:
        recon = t.t_upload + sum(s.t_queue for s in t.stages) + sum(s.t_compute for s in t.stages) + t.t_a2a
        np.testing.assert_allclose(t.t_total, recon, rtol=1e-9, atol=1e-9)
        assert t.t_a2a == 0.0
        assert t.stages[0].uav_id == 0
        assert t.stages[1].uav_id == 1
        assert t.stages[2].uav_id == 2


# ================================================================
# RUNNER
# ================================================================

def run_all():
    tests = [
        test_chain_creation,
        test_chain_ordering,
        test_each_stage_executes_once,
        test_correct_uav_placement,
        test_stage_persistence_across_slots,
        test_no_premature_completion,
        test_correct_stage_latency,
        test_total_latency_identity_phase3,
        test_computation_energy_all_stages,
        test_chain_metrics,
        test_chain_percentiles,
        test_deterministic_smoke_phase3,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
            traceback.print_exc()
    if failed:
        print(f"\n{len(failed)}/{len(tests)} FAILED:")
        for name, exc in failed:
            print(f"  FAIL  {name}: {exc}")
        return 1
    print(f"\nAll {len(tests)} Phase-3 tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())

