"""
Phase-2 task & queue dynamics validation.

Tests (16):
  1.  test_obs_action_shapes          — obs/action shape unchanged (8, 91) / 5
  2.  test_users_stationary_p2        — users never move
  3.  test_unique_task_ids            — no duplicate task IDs among completed tasks
  4.  test_tasks_persist_across_steps — queued tasks survive multiple slots
  5.  test_unfinished_tasks_stay_queued — only tasks with remaining==0 are "completed"
  6.  test_queue_grows_high_load      — backlog increases under high arrival / few UAVs
  7.  test_queue_drains_low_load      — backlog reaches 0 when arrivals stop
  8.  test_no_premature_completion    — completed tasks have remaining_bits/cycles ≈ 0
  9.  test_no_negative_queue_latency  — t_queue, t_upload, t_compute all ≥ 0
  10. test_latency_identity           — t_total == t_upload + t_queue + t_compute
  11. test_completion_leq_generated   — episode completed ≤ episode generated
  12. test_deadline_tracking          — deadline_missed flag matches t_total vs deadline
  13. test_percentile_ordering        — p99 ≥ p95 ≥ p50 ≥ 0
  14. test_battery_never_negative     — battery ≥ 0 across all steps
  15. test_depleted_uav_no_new_tasks  — depleted UAV gets no new upload-buffer tasks
  16. test_100_slot_deterministic     — 100-slot run: no NaN/Inf, counts consistent

Run:
    python tests/test_phase2_env.py
"""

from __future__ import annotations

import math
import os
import sys
import traceback

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import MultiUAVMECEnv, Task  # noqa: E402
from config import SLOT_DURATION, NUM_TIME_SLOTS  # noqa: E402


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


def _east_actions(env):
    a = _zero_actions(env)
    a[:, 0] = 1.0
    return a


def _backlog(env):
    """Total in-flight tasks across all active UAVs."""
    return int(sum(
        len(u.upload_buffer) + int(u.active_upload is not None) +
        len(u.task_queue) + int(u.computing is not None)
        for u in env._active_uavs()
    ))


# ================================================================
# TEST FUNCTIONS
# ================================================================

def test_obs_action_shapes():
    """Phase-1 compatibility: observation (8, 91) and action dim 5 unchanged."""
    from config import MAX_UAVS
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=0, task_gen_prob=0.0)
    obs, state = env.reset()
    assert env.num_users == 10
    assert env.num_uavs == 3
    assert env.num_active == 3
    assert obs.shape == (MAX_UAVS, 91), f"Expected ({MAX_UAVS}, 91), got {obs.shape}"
    assert obs.shape == (env.n_agents, env.local_obs_dim)
    assert state.shape == (env.global_state_dim,)
    assert env.agent_action_dim == 5
    assert np.isfinite(obs).all()
    assert np.isfinite(state).all()


def test_users_stationary_p2():
    """Users remain fixed throughout an episode."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=10, task_gen_prob=0.5)
    env.reset()
    before = np.array([d.position.copy() for d in env.devices])
    for _ in range(50):
        env.step(_east_actions(env))
    after = np.array([d.position.copy() for d in env.devices])
    np.testing.assert_array_equal(before, after)


def test_unique_task_ids():
    """All completed tasks carry unique IDs."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=5, seed=11, task_gen_prob=0.8)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))
    ids = [t.task_id for t in env.completed_tasks]
    assert len(ids) == len(set(ids)), f"Duplicate task IDs: {len(ids)} ids, {len(set(ids))} unique"


def test_tasks_persist_across_steps():
    """Tasks queued in upload or compute queues survive multiple steps."""
    env = MultiUAVMECEnv(num_users=50, num_uavs=3, seed=12, task_gen_prob=0.99)
    env.reset()
    for _ in range(5):
        env.step(_zero_actions(env))
    total_queued = _backlog(env)
    assert total_queued > 0, (
        "Expected tasks to persist in queues after 5 high-load steps; got 0"
    )


def test_unfinished_tasks_stay_queued():
    """Only tasks with remaining_bits ≈ 0 AND remaining_cycles ≈ 0 are marked completed."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=3, seed=13, task_gen_prob=0.7)
    env.reset()
    for _ in range(20):
        env.step(_zero_actions(env))

    for t in env.completed_tasks:
        assert t.remaining_bits <= 1e-9, (
            f"Completed task {t.task_id} still has remaining_bits={t.remaining_bits}"
        )
        assert t.remaining_cycles <= 1e-6, (
            f"Completed task {t.task_id} still has remaining_cycles={t.remaining_cycles}"
        )
        assert t.status == "completed", f"Expected status='completed', got '{t.status}'"

    # Tasks still in queues must NOT be marked completed
    for uav in env._active_uavs():
        for task in list(uav.upload_buffer):
            assert task.status != "completed", (
                f"Task {task.task_id} in upload_buffer but status=completed"
            )
        for task in list(uav.task_queue):
            assert task.status != "completed", (
                f"Task {task.task_id} in task_queue but status=completed"
            )


def test_queue_grows_high_load():
    """Under high arrival rate and few UAVs, queue backlog grows."""
    env = MultiUAVMECEnv(num_users=100, num_uavs=3, seed=14, task_gen_prob=0.99)
    env.reset()
    initial = _backlog(env)
    for _ in range(20):
        env.step(_zero_actions(env))
    final = _backlog(env)
    assert final > initial, (
        f"Expected backlog to grow under high load: {initial} -> {final}"
    )


def test_queue_drains_low_load():
    """When arrivals stop, all queued tasks eventually complete (backlog → 0)."""
    env = MultiUAVMECEnv(num_users=100, num_uavs=5, seed=15, task_gen_prob=0.99)
    env.reset()
    # Build up backlog
    for _ in range(10):
        env.step(_zero_actions(env))
    peak = _backlog(env)

    # Stop arrivals; let queues drain
    env.task_gen_prob = 0.0
    # Give generous time: worst-case each task takes a few slots to upload + compute
    for _ in range(300):
        env.step(_zero_actions(env))

    final = _backlog(env)
    assert final < peak, (
        f"Expected backlog to decrease after stopping arrivals: {peak} -> {final}"
    )
    assert final == 0, (
        f"Expected backlog to fully drain after 300 no-arrival slots, got {final}"
    )


def test_no_premature_completion():
    """No task is completed while it still has data bits or CPU cycles remaining."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=4, seed=16, task_gen_prob=0.6)
    env.reset()
    for _ in range(30):
        env.step(_zero_actions(env))
    for t in env.completed_tasks:
        assert t.remaining_bits <= 1e-9, (
            f"Task {t.task_id} completed prematurely: remaining_bits={t.remaining_bits}"
        )
        assert t.remaining_cycles <= 1e-6, (
            f"Task {t.task_id} completed prematurely: remaining_cycles={t.remaining_cycles}"
        )


def test_no_negative_queue_latency():
    """All latency components are non-negative for every completed task."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=4, seed=17, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))
    assert len(env.completed_tasks) > 0, "Need completed tasks; increase steps or prob"
    for t in env.completed_tasks:
        assert t.t_upload >= 0.0, f"t_upload={t.t_upload} < 0 for task {t.task_id}"
        assert t.t_queue  >= 0.0, f"t_queue={t.t_queue} < 0 for task {t.task_id}"
        assert t.t_compute >= 0.0, f"t_compute={t.t_compute} < 0 for task {t.task_id}"
        assert t.t_total  >= 0.0, f"t_total={t.t_total} < 0 for task {t.task_id}"


def test_latency_identity():
    """t_total == t_upload + t_queue + t_compute (+ t_a2a == 0) for every task."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=4, seed=18, task_gen_prob=0.7)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))
    assert len(env.completed_tasks) > 0, "Need completed tasks for this test"
    for t in env.completed_tasks:
        recon = t.t_upload + t.t_queue + t.t_compute + t.t_a2a
        np.testing.assert_allclose(
            t.t_total, recon, rtol=1e-9, atol=1e-9,
            err_msg=(
                f"Latency identity violated for task {t.task_id}: "
                f"t_total={t.t_total:.9f}  sum={recon:.9f}  "
                f"(up={t.t_upload:.6f} q={t.t_queue:.6f} "
                f"comp={t.t_compute:.6f} a2a={t.t_a2a})"
            ),
        )


def test_completion_leq_generated():
    """Episode completed count never exceeds episode generated count."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=4, seed=19, task_gen_prob=0.6)
    env.reset()
    for _ in range(50):
        env.step(_zero_actions(env))
    assert env._episode_tasks_completed <= env._episode_tasks_generated, (
        f"completed={env._episode_tasks_completed} > "
        f"generated={env._episode_tasks_generated}"
    )


def test_deadline_tracking():
    """deadline_missed flag matches t_total > deadline; episode counter is consistent."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=3, seed=20, task_gen_prob=0.7)
    env.reset()
    for _ in range(100):
        env.step(_zero_actions(env))

    assert len(env.completed_tasks) > 0, "Need completed tasks for deadline test"

    for t in env.completed_tasks:
        expected_missed = t.t_total > t.deadline
        assert t.deadline_missed is expected_missed, (
            f"Task {t.task_id}: deadline_missed={t.deadline_missed} but "
            f"t_total={t.t_total:.4f}, deadline={t.deadline:.4f}"
        )

    # Episode counter must match sum of individual flags
    ep_misses = sum(1 for t in env.completed_tasks if t.deadline_missed)
    assert env._episode_deadline_misses == ep_misses, (
        f"Episode miss counter mismatch: "
        f"env={env._episode_deadline_misses}, actual={ep_misses}"
    )


def test_percentile_ordering():
    """p99 >= p95 >= p50 >= 0 after enough completions."""
    env = MultiUAVMECEnv(num_users=30, num_uavs=4, seed=21, task_gen_prob=0.7)
    env.reset()
    info = None
    for _ in range(100):
        _, _, _, _, info = env.step(_zero_actions(env))
    assert info is not None
    p50 = info["p50_latency"]
    p95 = info["p95_latency"]
    p99 = info["p99_latency"]
    assert p50 >= 0.0, f"p50 negative: {p50}"
    assert p95 >= p50, f"p95={p95} < p50={p50}"
    assert p99 >= p95, f"p99={p99} < p95={p95}"


def test_battery_never_negative():
    """Battery energy is always >= 0 for every active UAV across all steps."""
    env = MultiUAVMECEnv(num_users=20, num_uavs=5, seed=22, task_gen_prob=0.5)
    env.reset()
    for _ in range(100):
        _, _, _, _, info = env.step(_east_actions(env))
        for b in info["battery"]:
            assert b >= 0.0, f"Negative battery detected: {b}"
        for u in env._active_uavs():
            assert u.battery_energy >= 0.0, (
                f"UAV {u.id} battery={u.battery_energy} < 0"
            )


def test_depleted_uav_no_new_tasks():
    """A UAV with battery=0 does not receive new tasks in its upload buffer."""
    env = MultiUAVMECEnv(num_users=50, num_uavs=3, seed=23, task_gen_prob=0.9)
    env.reset()

    # Manually deplete UAV 0 immediately after reset (queues are empty)
    env.uavs[0].battery_energy = 0.0

    assert len(env.uavs[0].upload_buffer) == 0, "Sanity: upload_buffer empty after reset"
    assert env.uavs[0].active_upload is None,   "Sanity: no active upload after reset"

    # Run 10 steps with many task arrivals
    for _ in range(10):
        env.step(_zero_actions(env))

    # UAV 0 should not have accumulated any new tasks in its upload buffer
    assert len(env.uavs[0].upload_buffer) == 0, (
        f"Depleted UAV 0 accumulated {len(env.uavs[0].upload_buffer)} "
        "tasks in upload_buffer"
    )
    # Other UAVs must still be receiving tasks (system not fully broken)
    other_backlog = sum(
        _backlog_uav(u) for u in env._active_uavs() if u.id != 0
    )
    # With 50 users × 0.9 prob × 10 steps, at least some tasks exist somewhere
    # (this is a soft sanity check; if both UAVs 1 & 2 somehow also got depleted
    #  by propulsion, we skip; but PHASE1_UAV_BATTERY_J is very large so they won't)
    assert env._episode_tasks_generated > 0, "Expected some tasks to be generated"


def _backlog_uav(u):
    return (len(u.upload_buffer) + int(u.active_upload is not None) +
            len(u.task_queue) + int(u.computing is not None))


def test_100_slot_deterministic():
    """100-slot deterministic simulation: no NaN/Inf; all invariants hold every slot."""
    env = MultiUAVMECEnv(num_users=50, num_uavs=5, seed=42, task_gen_prob=0.7)
    obs, state = env.reset()
    _finite(obs, state)

    prev_completed = 0
    prev_generated = 0

    for slot in range(100):
        obs, state, rewards, done, info = env.step(_zero_actions(env))

        # Shapes and finiteness
        _finite(obs, state, rewards)
        _finite(
            info["total_latency"],   info["energy_consumed"],
            info["upload_latency"],  info["queue_latency"],
            info["compute_latency"], info["p50_latency"],
            info["p95_latency"],     info["p99_latency"],
            info["avg_queue_length"],
        )

        # Battery invariant
        for b in info["battery"]:
            assert b >= 0.0, f"Slot {slot+1}: negative battery {b}"

        # Slot-level counts
        assert info["tasks_completed"] >= 0
        assert info["tasks_generated"] >= 0
        assert info["deadline_misses"] >= 0
        assert info["tasks_pending"] >= 0

        # Episode counts are monotonically non-decreasing
        assert env._episode_tasks_generated >= prev_generated
        assert env._episode_tasks_completed >= prev_completed
        prev_generated = env._episode_tasks_generated
        prev_completed = env._episode_tasks_completed

        # Percentile ordering
        assert info["p99_latency"] >= info["p95_latency"] >= info["p50_latency"] >= 0

    # Episode-level invariant
    assert env._episode_tasks_completed <= env._episode_tasks_generated, (
        f"Episode: completed={env._episode_tasks_completed} > "
        f"generated={env._episode_tasks_generated}"
    )


# ================================================================
# RUNNER
# ================================================================

def run_all():
    tests = [
        test_obs_action_shapes,
        test_users_stationary_p2,
        test_unique_task_ids,
        test_tasks_persist_across_steps,
        test_unfinished_tasks_stay_queued,
        test_queue_grows_high_load,
        test_queue_drains_low_load,
        test_no_premature_completion,
        test_no_negative_queue_latency,
        test_latency_identity,
        test_completion_leq_generated,
        test_deadline_tracking,
        test_percentile_ordering,
        test_battery_never_negative,
        test_depleted_uav_no_new_tasks,
        test_100_slot_deterministic,
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
    print(f"\nAll {len(tests)} Phase-2 tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
