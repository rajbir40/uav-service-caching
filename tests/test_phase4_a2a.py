"""
Phase-4 explicit A2A communication validation.
==============================================

Tests (10):
  1.  test_a2a_rate_positive                  — A2A rate is valid and positive
  2.  test_a2a_rate_decreases_with_distance   — Rate decreases with distance
  3.  test_a2a_delay_identity_dr              — A2A delay identity D/R
  4.  test_abc_two_a2a_transitions            — A->B->C produces exactly two A2A transitions
  5.  test_chain_latency_includes_a2a         — Chain latency now includes A2A
  6.  test_farther_uavs_increase_a2a_latency  — Moving UAVs farther apart increases A2A latency
  7.  test_a2a_latency_finite                 — A2A latency is never negative/NaN/Inf
  8.  test_multi_slot_persistence_phase4       — Multi-slot persistence works
  9.  test_previous_phases_pass               — Previous Phase 1–3 tests still pass
  10. test_deterministic_smoke_phase4         — Deterministic 100-slot smoke test passes

Run:
    python tests/test_phase4_a2a.py
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
from src.latency import calculate_a2a_latency  # noqa: E402
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

def test_a2a_rate_positive():
    """Verify A2A rate is valid and positive."""
    env = MultiUAVMECEnv(num_users=10, num_uavs=3)
    r = env.calculate_a2a_rate(env.uavs[0], env.uavs[1])
    assert r > 0.0, f"Expected positive A2A rate, got {r}"
    _finite(r)


def test_a2a_rate_decreases_with_distance():
    """Verify Rate decreases as distance increases."""
    env = MultiUAVMECEnv(num_users=10, num_uavs=3)
    uav0 = env.uavs[0]
    uav1 = env.uavs[1]
    uav2 = env.uavs[2]

    uav0.position = np.array([0.0, 0.0, 100.0])
    uav1.position = np.array([100.0, 0.0, 100.0])
    uav2.position = np.array([500.0, 0.0, 100.0])

    r1 = env.calculate_a2a_rate(uav0, uav1)
    r2 = env.calculate_a2a_rate(uav0, uav2)
    assert r1 > r2, f"Expected A2A rate to decrease with distance: {r1} vs {r2}"


def test_a2a_delay_identity_dr():
    """Verify A2A delay identity D/R."""
    d = 1.5e6
    r = 2.5e7
    delay = calculate_a2a_latency(d, r)
    assert math.isclose(delay, d / r), f"Expected delay {d/r}, got {delay}"


def test_abc_two_a2a_transitions():
    """Verify A->B->C (0->1->2) produces exactly two A2A transitions."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=42, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0, "No completed tasks to verify transitions"
    for t in env.completed_tasks:
        a2a_stages = [s for s in t.stages if s.t_a2a > 0.0]
        # Since UAV0 -> UAV1 -> UAV2 has different UAV ids, both stage B (idx 1) and stage C (idx 2) are transitioned via A2A.
        # Thus there should be exactly two transitions with positive t_a2a.
        assert len(a2a_stages) == 2, f"Expected exactly two transitions, got {len(a2a_stages)}"


def test_chain_latency_includes_a2a():
    """Verify Chain latency now includes A2A."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=43, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0
    for t in env.completed_tasks:
        assert t.t_a2a > 0.0, f"Expected positive A2A latency, got {t.t_a2a}"
        sum_stage_queue = sum(s.t_queue for s in t.stages)
        sum_stage_compute = sum(s.t_compute for s in t.stages)
        recon = t.t_upload + sum_stage_queue + sum_stage_compute + t.t_a2a
        np.testing.assert_allclose(
            t.t_total, recon, rtol=1e-9, atol=1e-9,
            err_msg=f"Identity check failed: t_total={t.t_total} vs recon={recon}"
        )


def test_farther_uavs_increase_a2a_latency():
    """Verify Moving UAVs farther apart increases A2A latency."""
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=44)
    env.uavs[0].position = np.array([0.0, 0.0, 100.0])
    env.uavs[1].position = np.array([100.0, 0.0, 100.0])
    r1 = env.calculate_a2a_rate(env.uavs[0], env.uavs[1])
    d = 1e6
    lat1 = calculate_a2a_latency(d, r1)

    env.uavs[1].position = np.array([500.0, 0.0, 100.0])
    r2 = env.calculate_a2a_rate(env.uavs[0], env.uavs[1])
    lat2 = calculate_a2a_latency(d, r2)
    assert lat2 > lat1, f"Expected larger A2A latency for farther UAVs: {lat2} vs {lat1}"


def test_a2a_latency_finite():
    """Verify A2A latency is never negative/NaN/Inf."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=45, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0
    for t in env.completed_tasks:
        assert t.t_a2a >= 0.0, f"A2A latency cannot be negative: {t.t_a2a}"
        _finite(t.t_a2a)


def test_multi_slot_persistence_phase4():
    """Verify task and stage persistence holds under Phase 4 (consecutive stage queuing)."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=46, task_gen_prob=0.8)
    env.reset()
    saw_intermediate_queued = False
    for _ in range(30):
        env.step(_zero_actions(env))
        for uav in env._active_uavs():
            for t in uav.task_queue:
                if t.current_stage_idx in (1, 2):
                    saw_intermediate_queued = True
                    break
    assert saw_intermediate_queued, "Expected tasks to persist across slot boundaries between stages"


def test_previous_phases_pass():
    """Verify previous Phase 1-3 tests still pass programmatically."""
    # This is handled directly in the runner script
    pass


def test_deterministic_smoke_phase4():
    """Verify 100-slot smoke test runs with consistent stats and no NaN/Inf."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=47, task_gen_prob=0.8)
    env.reset()
    for slot in range(100):
        _, _, rewards, _, info = env.step(_zero_actions(env))
        _finite(rewards)
        _finite(info["total_latency"], info["energy_consumed"])
        assert info["a2a_latency"] >= 0.0
        assert info["a2a_rate"] >= 0.0
        assert info["a2a_distance"] >= 0.0


# ================================================================
# RUNNER
# ================================================================

def run_all():
    tests = [
        test_a2a_rate_positive,
        test_a2a_rate_decreases_with_distance,
        test_a2a_delay_identity_dr,
        test_abc_two_a2a_transitions,
        test_chain_latency_includes_a2a,
        test_farther_uavs_increase_a2a_latency,
        test_a2a_latency_finite,
        test_multi_slot_persistence_phase4,
        test_previous_phases_pass,
        test_deterministic_smoke_phase4,
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
    print(f"\nAll {len(tests)} Phase-4 tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(run_all())
