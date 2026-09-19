"""
Phase-1 environment validation.

Run:
    python tests/test_phase1_env.py
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.env import MultiUAVMECEnv  # noqa: E402
from src.channel_model import ChannelModel, NOMACommunication  # noqa: E402
from src.latency import (  # noqa: E402
    calculate_a2g_distance,
    calculate_upload_latency,
    calculate_compute_latency,
    calculate_a2a_latency,
    calculate_total_latency,
)
from config import SLOT_DURATION, NOISE_PSD, IOT_TX_POWER  # noqa: E402


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


def test_obs_action_shapes():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=0, task_gen_prob=0.0)
    obs, state = env.reset()
    assert env.num_users == 10
    assert env.num_uavs == 3
    assert env.num_active == 3
    assert obs.shape == (env.n_agents, env.local_obs_dim)
    assert state.shape == (env.global_state_dim,)
    assert env.agent_action_dim == 5
    assert np.isfinite(obs).all()
    assert np.isfinite(state).all()


def test_user_positions_frozen():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=1, task_gen_prob=0.5)
    env.reset()
    before = np.array([d.position.copy() for d in env.devices])
    for _ in range(20):
        env.step(_east_actions(env))
    after = np.array([d.position.copy() for d in env.devices])
    np.testing.assert_array_equal(before, after)


def test_uav_moves_with_nonzero_action():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=2, task_gen_prob=0.0)
    env.reset()
    before = env.uavs[0].position.copy()
    env.step(_east_actions(env))
    after = env.uavs[0].position
    assert after[0] != before[0] or after[1] != before[1], "UAV should move"


def test_a2g_distance_changes_with_uav():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=3, task_gen_prob=0.0)
    env.reset()
    user = env.devices[0]
    d0 = env.calculate_a2g_distance(user.position, env.uavs[0].position)
    env.step(_east_actions(env))
    d1 = env.calculate_a2g_distance(user.position, env.uavs[0].position)
    assert d0 != d1


def test_rate_decreases_with_distance():
    ch = ChannelModel()
    ch.wind_speed = 0.0
    user = np.array([0.0, 0.0, 0.0])
    near = np.array([50.0, 0.0, 100.0])
    far = np.array([400.0, 0.0, 100.0])
    d_near = calculate_a2g_distance(user, near)
    d_far = calculate_a2g_distance(user, far)
    assert d_far > d_near
    bw = 10e6
    r_near = ch.a2g_uplink_rate(user, near, IOT_TX_POWER, bw, NOISE_PSD)
    r_far = ch.a2g_uplink_rate(user, far, IOT_TX_POWER, bw, NOISE_PSD)
    assert r_near > r_far > 0
    t_near = calculate_upload_latency(2e6, r_near)
    t_far = calculate_upload_latency(2e6, r_far)
    assert t_far > t_near > 0


def test_compute_latency_scales_with_cycles():
    f = 2e9
    t_small = calculate_compute_latency(1e8, f)
    t_large = calculate_compute_latency(8e8, f)
    assert t_large > t_small > 0
    np.testing.assert_allclose(t_large / t_small, 8.0, rtol=1e-9)


def test_energy_and_battery():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=4, task_gen_prob=0.3)
    env.reset()
    batteries = [u.battery_energy for u in env.uavs]
    _, _, _, _, info0 = env.step(_zero_actions(env))
    hover_e = info0["energy_flight"]
    env.reset()
    _, _, _, _, info1 = env.step(_east_actions(env))
    move_e = info1["energy_flight"]
    assert move_e >= hover_e
    assert info1["energy_consumed"] > 0
    for u, b0 in zip(env.uavs, batteries):
        assert u.battery_energy <= b0
        assert u.battery_energy >= 0.0
    env2 = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=4, task_gen_prob=0.3)
    env2.reset()
    prev = [u.battery_energy for u in env2.uavs]
    for _ in range(20):
        env2.step(_east_actions(env2))
        for u, p in zip(env2.uavs, prev):
            assert u.battery_energy <= p + 1e-9
            assert u.battery_energy >= 0.0
        prev = [u.battery_energy for u in env2.uavs]


def test_total_latency_identity():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=5, task_gen_prob=0.8)
    env.reset()
    for _ in range(30):
        env.step(_zero_actions(env))
    assert len(env.completed_tasks) > 0, "expected some completed tasks"
    for task in env.completed_tasks:
        recon = calculate_total_latency(task.t_upload, task.t_queue, task.t_compute, task.t_a2a)
        np.testing.assert_allclose(task.t_total, recon, rtol=1e-9, atol=1e-9)
        assert task.t_upload >= 0 and task.t_queue >= 0 and task.t_compute >= 0
        assert task.t_a2a == 0.0


def test_no_noma_sic_jammer_wind_a2a():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=6, task_gen_prob=0.5)
    env.reset()
    assert not hasattr(env, "noma")
    assert env.jammers == []
    assert env.wind_speed == 0.0

    called = {"noma": False}

    orig = NOMACommunication.serve_cluster

    def _guard(self, *args, **kwargs):
        called["noma"] = True
        return orig(self, *args, **kwargs)

    NOMACommunication.serve_cluster = _guard
    try:
        for _ in range(20):
            _, _, rewards, _, info = env.step(_east_actions(env))
            assert info["a2a_latency"] == 0.0
            assert info["noma_pairs_formed"] == 0
            assert info["jammer_blocked"] == 0
            assert info["wind_speed"] == 0.0
            assert calculate_a2a_latency() == 0.0
            _finite(rewards, info["total_latency"], info["energy_consumed"],
                    info["upload_latency"], info["queue_latency"],
                    info["compute_latency"])
            assert np.all(np.asarray(info["battery"]) >= 0.0)
    finally:
        NOMACommunication.serve_cluster = orig
    assert called["noma"] is False


def test_smoke_print():
    env = MultiUAVMECEnv(num_users=10, num_uavs=3, seed=7, task_gen_prob=0.6)
    obs, state = env.reset()
    assert env.num_users == 10 and env.num_uavs == 3
    print("\n=== Phase-1 smoke test (3 UAVs, 10 users, 20 slots) ===")
    print(f"obs_dim={env.local_obs_dim}  state_dim={env.global_state_dim}  "
          f"action_dim={env.agent_action_dim}")
    print(f"user0={env.devices[0].position.tolist()}")
    for slot in range(20):
        obs, state, rewards, done, info = env.step(_east_actions(env))
        uav_pos = [np.round(u.position, 2).tolist() for u in env.uavs]
        print(
            f"slot={slot+1:02d}  gen={info['num_tasks_generated']}  "
            f"done={info['num_tasks_completed']}  "
            f"d={info['a2g_distance']:.1f}  R={info['a2g_rate']:.2e}  "
            f"Tup={info['upload_latency']:.4f}  Tq={info['queue_latency']:.4f}  "
            f"Tc={info['compute_latency']:.4f}  T={info['total_latency']:.4f}  "
            f"E={info['energy_consumed']:.1f}  "
            f"bat={[round(b, 1) for b in info['battery']]}  "
            f"q={info['queue_length']}  r={float(np.mean(rewards[:env.num_active])):.4f}"
        )
        _finite(obs, state, rewards)
        assert done is False
        assert all(b >= 0 for b in info["battery"])
    print(f"uav_pos={uav_pos}")
    print(f"user0 after={env.devices[0].position.tolist()}")
    print("=== smoke test finished ===\n")


def run_all():
    tests = [
        test_obs_action_shapes,
        test_user_positions_frozen,
        test_uav_moves_with_nonzero_action,
        test_a2g_distance_changes_with_uav,
        test_rate_decreases_with_distance,
        test_compute_latency_scales_with_cycles,
        test_energy_and_battery,
        test_total_latency_identity,
        test_no_noma_sic_jammer_wind_a2a,
        test_smoke_print,
    ]
    failed = []
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
        except Exception as exc:
            failed.append((fn.__name__, exc))
            print(f"FAIL  {fn.__name__}: {exc}")
            raise
    print(f"\nAll {len(tests)} Phase-1 tests passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run_all())
