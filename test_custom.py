"""
test_custom.py — Test a trained Diffusion-MAPPO model on a custom environment
==============================================================================

Auto-selects the best available checkpoint in this priority order:
    1. results/checkpoints/best_model.pt   (highest reward during training)
    2. results/checkpoints/final_model.pt  (saved at end of training)
    3. results/checkpoints/model_step*.pt  (highest step number found)

You can also override with --checkpoint to specify a path manually.

Outputs saved to:  results/test_results/<checkpoint_name>_<timestamp>/
    logs/
        training_metrics.jsonl  — per-step metrics (same format as training)
        eval_sweeps.json        — parameter sweep results for P2-P11
    trajectories/
        flight_path_0000001.npz — UAV + IoT trajectory data for P8
    plots/
        P1a_reward.*            — reward convergence curve
        P1b_latency.*           — latency convergence curve
        P2_latency_vs_N.*       — latency vs number of IoT devices
        P3_latency_vs_M.*       — latency vs number of UAVs
        P4_latency_vs_J.*       — latency vs number of jammers
        P5_energy.*             — cumulative UAV energy
        P6_latency_vs_cache.*   — latency vs cache capacity
        P7_cache_vs_zipf.*      — cache hit rate vs Zipf exponent
        P8_traj_2d.*            — 2D UAV trajectory plot
        P8_traj_3d.*            — 3D UAV trajectory plot
        P9_latency_vs_jammer.*  — latency vs jammer power
        P10_latency_vs_task.*   — latency vs task size
        P11_latency_vs_tx.*     — latency vs IoT TX power
        summary.png             — 4-panel episode metric summary
    metrics.json                — summary stats (mean +/- std)
    episodes.jsonl              — per-episode results

Usage examples:
    # Auto-select best/final checkpoint, default env params
    python test_custom.py

    # Custom environment parameters
    python test_custom.py --num-iot 120 --num-uavs 6 --num-jammers 2

    # Full custom test
    python test_custom.py \
        --num-iot 130 --num-uavs 7 --num-jammers 3 \
        --area-radius 550 --episodes 20

    # Manually specify a checkpoint
    python test_custom.py --checkpoint results/checkpoints/model_step1000000.pt \
        --num-uavs 4 --num-jammers 1

    # Run on GPU
    python test_custom.py --device cuda --num-uavs 6
"""

import sys
import os
import glob
import argparse
import re
import json
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

from config import *
from src.env import MultiUAVMECEnv
from src.mappo_agent import MAPPOAgent


# ================================================================
# PLOT STYLE
# ================================================================

plt.rcParams.update({
    'font.family'      : 'serif',
    'font.size'        : 9,
    'axes.titlesize'   : 9,
    'axes.labelsize'   : 9,
    'xtick.labelsize'  : 8,
    'ytick.labelsize'  : 8,
    'legend.fontsize'  : 8,
    'figure.dpi'       : 150,
    'axes.grid'        : True,
    'grid.linewidth'   : 0.3,
    'grid.alpha'       : 0.45,
    'axes.spines.top'  : False,
    'axes.spines.right': False,
})

COLORS = {
    'reward' : '#1A6FBF',
    'latency': '#E07020',
    'tx'     : '#2E8B57',
    'noma'   : '#9C27B0',
    'energy' : '#B03A2E',
}


# ================================================================
# AUTO CHECKPOINT SELECTION
# ================================================================

def find_best_checkpoint(checkpoint_dir='results/checkpoints'):
    """
    Auto-select checkpoint in priority order:
      1. best_model.pt   — best reward during training
      2. final_model.pt  — end-of-training save
      3. model_step*.pt  — highest step number found
    Returns (path, label) or (None, None).
    """
    best_path  = os.path.join(checkpoint_dir, 'best_model.pt')
    final_path = os.path.join(checkpoint_dir, 'final_model.pt')

    if os.path.exists(best_path):
        return best_path, 'best_model  (highest reward during training)'

    if os.path.exists(final_path):
        return final_path, 'final_model  (end of training)'

    step_files = glob.glob(os.path.join(checkpoint_dir, 'model_step*.pt'))
    if step_files:
        def _step(p):
            m = re.search(r'model_step(\d+)\.pt', p)
            return int(m.group(1)) if m else 0
        step_files.sort(key=_step)
        chosen = step_files[-1]
        return chosen, f'model_step{_step(chosen)}.pt  (latest periodic checkpoint)'

    return None, None


# ================================================================
# TRAJECTORY CAPTURE  (uses real trained agent, not random actions)
# ================================================================

def capture_trajectory(env, agent, traj_dir, step_tag=1):
    """
    Run one full episode with the trained agent and save a flight_path .npz
    to traj_dir — exactly the same format that logger.save_trajectory_episode()
    produces, so plot_P8 can read it directly.
    """
    os.makedirs(traj_dir, exist_ok=True)

    # Throw-away env so we don't disturb the main test env
    tmp_env = MultiUAVMECEnv(
        num_users  = env.num_users,
        num_uavs   = env.num_uavs,
        num_jammers= env.num_jammers,
        seed       = step_tag,
        domain_cfg = env.domain_cfg,
    )
    obs, state = tmp_env.reset()

    # Snapshot positions at episode start
    dev_pos_start  = np.array([d.position.copy() for d in tmp_env.devices])
    dev_cluster    = np.array([d.cluster_id for d in tmp_env.devices])
    dev_jammed     = np.array([tmp_env.is_in_jammer_zone(d) for d in tmp_env.devices])
    uav_pos_start  = np.array([u.position.copy() for u in tmp_env.uavs[:tmp_env.num_active]])

    uav_trajs = {i: [] for i in range(tmp_env.num_active)}
    iot_trajs = {i: [tmp_env.devices[i].position[:2].copy()]
                 for i in range(len(tmp_env.devices))}

    noma_snapshot = None
    coop_snapshot = None

    for _ in range(VIDEO_PLOT_STEPS):
        with torch.no_grad():
            actions, _, _ = agent.get_actions(
                obs, state, num_active=tmp_env.num_active)
        obs, state, _, done, info = tmp_env.step(actions)

        for i in range(tmp_env.num_active):
            uav_trajs[i].append(tmp_env.uavs[i].position[:2].copy())
        for i, dev in enumerate(tmp_env.devices):
            iot_trajs[i].append(dev.position[:2].copy())

        # Capture first NOMA pair
        if noma_snapshot is None and info.get('served_dev_ids'):
            agent_devs = {}
            for dev_id, entry in info['served_dev_ids'].items():
                aid = entry['agent']
                if entry.get('noma', False):
                    agent_devs.setdefault(aid, []).append(dev_id)
            for aid, dev_ids in agent_devs.items():
                if len(dev_ids) >= 2:
                    uav_xy = tmp_env.uavs[aid].position[:2].copy()
                    d0 = tmp_env.devices[dev_ids[0]].position[:2].copy()
                    d1 = tmp_env.devices[dev_ids[1]].position[:2].copy()
                    if np.linalg.norm(d0 - uav_xy) < np.linalg.norm(d1 - uav_xy):
                        d0, d1 = d1, d0
                    noma_snapshot = (aid, uav_xy, np.array([d0, d1]))
                    break

        # Capture first cooperative offload
        if coop_snapshot is None and info.get('offload_links'):
            for src_id, dst_id in info['offload_links']:
                if 0 <= src_id < tmp_env.num_active and 0 <= dst_id < tmp_env.num_active:
                    coop_snapshot = (
                        src_id, dst_id,
                        tmp_env.uavs[src_id].position[:2].copy(),
                        tmp_env.uavs[dst_id].position[:2].copy(),
                    )
                    break

        if done:
            break

    # Build trajectory arrays
    traj_arrays = {f'uav_{i}': np.array(traj) for i, traj in uav_trajs.items()}
    MAX_IOT_TRAJ = 500
    iot_items = [(i, t) for i, t in iot_trajs.items() if len(t) >= 2]
    if len(iot_items) > MAX_IOT_TRAJ:
        stride = len(iot_items) // MAX_IOT_TRAJ
        iot_items = iot_items[::stride]
    for i, t in iot_items:
        traj_arrays[f'iot_{i}'] = np.array(t)

    dev_pos_end  = np.array([d.position.copy() for d in tmp_env.devices])
    jam_pos      = (np.array([j.position.copy() for j in tmp_env.jammers])
                    if tmp_env.jammers else np.zeros((0, 3)))

    COMM_RANGE = 200.0
    dev_in_range = np.ones(len(tmp_env.devices), dtype=bool)
    for idx, d in enumerate(tmp_env.devices):
        cid = d.cluster_id
        if 0 <= cid < tmp_env.num_active:
            dist = np.linalg.norm(d.position[:2] - tmp_env.uavs[cid].position[:2])
            dev_in_range[idx] = bool(dist <= COMM_RANGE)

    # NOMA pack
    if noma_snapshot is not None:
        noma_uav_id  = np.array([noma_snapshot[0]], dtype=np.int32)
        noma_uav_pos = noma_snapshot[1].astype(np.float32)
        noma_dev_pos = noma_snapshot[2].astype(np.float32)
    else:
        noma_uav_id  = np.array([-1], dtype=np.int32)
        noma_uav_pos = np.zeros(2, dtype=np.float32)
        noma_dev_pos = np.zeros((0, 2), dtype=np.float32)

    # Coop offload pack
    if coop_snapshot is not None:
        coop_src_id  = np.array([coop_snapshot[0]], dtype=np.int32)
        coop_dst_id  = np.array([coop_snapshot[1]], dtype=np.int32)
        coop_src_pos = coop_snapshot[2].astype(np.float32)
        coop_dst_pos = coop_snapshot[3].astype(np.float32)
    else:
        coop_src_id  = np.array([-1], dtype=np.int32)
        coop_dst_id  = np.array([-1], dtype=np.int32)
        coop_src_pos = np.zeros(2, dtype=np.float32)
        coop_dst_pos = np.zeros(2, dtype=np.float32)

    npz_path = os.path.join(traj_dir, f'flight_path_{step_tag:07d}.npz')
    np.savez_compressed(
        npz_path,
        step         = step_tag,
        area_radius  = tmp_env.area_radius,
        num_active   = tmp_env.num_active,
        jam_pos      = jam_pos,
        jam_radius   = tmp_env.jammer_interference_radius,
        uav_comm_range   = COMM_RANGE,
        uav_pos_start    = uav_pos_start,
        dev_pos_start    = dev_pos_start,
        dev_pos_end      = dev_pos_end,
        dev_cluster      = dev_cluster,
        dev_jammed       = dev_jammed,
        dev_in_range     = dev_in_range,
        wind_speed       = np.float32(tmp_env.wind_speed),
        wind_direction   = np.float32(tmp_env.wind_direction),
        noma_uav_id      = noma_uav_id,
        noma_uav_pos     = noma_uav_pos,
        noma_dev_pos     = noma_dev_pos,
        coop_src_id      = coop_src_id,
        coop_dst_id      = coop_dst_id,
        coop_src_pos     = coop_src_pos,
        coop_dst_pos     = coop_dst_pos,
        **traj_arrays,
    )
    return npz_path


# ================================================================
# RUN PARAMETER SWEEPS  (reuses logger's sweep runner)
# ================================================================

def run_sweeps(env, agent, log_dir):
    """
    Run the same parameter sweeps as end-of-training (P2-P11 data),
    save eval_sweeps.json to log_dir.
    """
    from src.logger import TrainingLogger
    import src.logger as logger_mod

    # Patch logger module to write to our test log_dir
    orig_log_dir  = logger_mod.LOG_DIR
    orig_traj_dir = logger_mod.TRAJ_DIR
    logger_mod.LOG_DIR  = log_dir
    logger_mod.TRAJ_DIR = log_dir   # sweeps don't write trajectories

    os.makedirs(log_dir, exist_ok=True)

    try:
        logger = TrainingLogger()
        logger.sweep_path = os.path.join(log_dir, 'eval_sweeps.json')
        logger.run_and_save_sweeps(MultiUAVMECEnv, n_episodes=3, n_steps=VIDEO_PLOT_STEPS)
    finally:
        # Always restore
        logger_mod.LOG_DIR  = orig_log_dir
        logger_mod.TRAJ_DIR = orig_traj_dir

    return os.path.join(log_dir, 'eval_sweeps.json')


# ================================================================
# WRITE TEST METRICS  (same .jsonl format as training logger)
# ================================================================

def write_test_metrics(step_log, log_dir):
    """Write per-step data as training_metrics.jsonl so P1/P5 can read it."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'training_metrics.jsonl')
    with open(path, 'w') as f:
        for row in step_log:
            f.write(json.dumps(row) + '\n')
    return path


# ================================================================
# GENERATE ALL IEEE PLOTS  (monkey-patch plot_ieee dirs then call all Px)
# ================================================================

def generate_all_plots(out_dir, log_dir, traj_dir):
    """
    Redirect plot_ieee's module-level directory constants to our test
    directories, then call every plot function so they all write to
    out_dir/plots/.
    """
    import plot_ieee as pie

    plots_dir = os.path.join(out_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    # Save original globals
    orig_log_dir      = pie.LOG_DIR
    orig_traj_dir     = pie.TRAJ_DIR
    orig_out_dir      = pie.OUT_DIR
    orig_traj_plot    = pie.TRAJ_PLOT_DIR

    # Redirect to test-local directories
    pie.LOG_DIR       = log_dir
    pie.TRAJ_DIR      = traj_dir
    pie.OUT_DIR       = plots_dir
    pie.TRAJ_PLOT_DIR = plots_dir

    generated = []
    skipped   = []

    try:
        plots = [
            ('P1  — Reward & Latency convergence',         pie.plot_P1),
            ('P2  — Latency vs IoT devices (N)',            pie.plot_P2),
            ('P3  — Latency vs UAVs (M)',                   pie.plot_P3),
            ('P4  — Latency vs Jammers (J)',                pie.plot_P4),
            ('P5  — Cumulative UAV energy',                 pie.plot_P5),
            ('P6  — Latency vs cache capacity',             pie.plot_P6),
            ('P7  — Cache hit rate vs Zipf exponent',       pie.plot_P7),
            ('P9  — Latency vs jammer power',               pie.plot_P9),
            ('P10 — Latency vs task size',                  pie.plot_P10),
            ('P11 — Latency vs IoT TX power',               pie.plot_P11),
        ]

        for label, fn in plots:
            try:
                before = set(glob.glob(os.path.join(plots_dir, '*')))
                fn()
                after  = set(glob.glob(os.path.join(plots_dir, '*')))
                new_files = after - before
                if new_files:
                    generated.extend(sorted(new_files))
                    print(f"    [OK]      {label}")
                else:
                    skipped.append(label)
                    print(f"    [skipped] {label}  (no data)")
            except Exception as e:
                skipped.append(label)
                print(f"    [error]   {label}: {e}")

        # P8 — 2D & 3D trajectory (separate call, needs output_dir)
        try:
            before = set(glob.glob(os.path.join(plots_dir, '*')))
            pie.plot_P8(step=None, output_dir=plots_dir,
                        wind_dir=None, wind_speed=None)
            after  = set(glob.glob(os.path.join(plots_dir, '*')))
            new_files = after - before
            if new_files:
                generated.extend(sorted(new_files))
                print(f"    [OK]      P8  — 2D & 3D UAV trajectory")
            else:
                skipped.append('P8')
                print(f"    [skipped] P8  — 2D & 3D UAV trajectory (no .npz data)")
        except Exception as e:
            skipped.append('P8')
            print(f"    [error]   P8 — 2D & 3D trajectory: {e}")

    finally:
        # Always restore original globals
        pie.LOG_DIR       = orig_log_dir
        pie.TRAJ_DIR      = orig_traj_dir
        pie.OUT_DIR       = orig_out_dir
        pie.TRAJ_PLOT_DIR = orig_traj_plot

    return plots_dir, generated, skipped


# ================================================================
# SUMMARY PLOT  (per-episode metrics, 4-panel)
# ================================================================

def _moving_average(values, window):
    if window <= 1:
        return np.array(values)
    kernel = np.ones(window) / window
    padded = np.pad(values, (window - 1, 0), mode='edge')
    return np.convolve(padded, kernel, mode='valid')


def generate_summary_plot(episode_data, plots_dir, env_info, ckpt_name):
    os.makedirs(plots_dir, exist_ok=True)
    rewards = [e['reward']  for e in episode_data]
    latency = [e['latency'] for e in episode_data]
    tx      = [e['tx']      for e in episode_data]
    noma    = [e['noma']    for e in episode_data]
    n_ep    = len(episode_data)
    ma_w    = max(1, min(5, n_ep // 3))
    x       = np.arange(1, n_ep + 1)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    fig.suptitle(
        f'Test Results — {ckpt_name}\n'
        f'IoT: {env_info["num_iot"]}  |  UAVs: {env_info["num_uavs"]}  |  '
        f'Jammers: {env_info["num_jammers"]}  |  '
        f'Area: {env_info["area_radius"]} m  |  {n_ep} episodes',
        fontsize=9, y=1.01,
    )
    panels = [
        (axes[0, 0], rewards, 'Reward',           'Mean Reward',       COLORS['reward']),
        (axes[0, 1], latency, 'Latency',           'Avg Latency (ms)',  COLORS['latency']),
        (axes[1, 0], tx,      'Successful TX',     'Avg Successful TX', COLORS['tx']),
        (axes[1, 1], noma,    'NOMA Pairs Formed', 'Avg NOMA Pairs',    COLORS['noma']),
    ]
    for ax, vals, title, ylabel, color in panels:
        ax.bar(x, vals, color=color, alpha=0.30, width=0.7)
        if len(vals) >= ma_w:
            ax.plot(x, _moving_average(vals, ma_w), color=color, lw=1.5)
        ax.axhline(np.mean(vals), color=color, lw=1.0, linestyle='--',
                   alpha=0.8, label=f'μ = {np.mean(vals):.2f}')
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel('Episode')
        ax.set_xlim(0.3, n_ep + 0.7)
        ax.xaxis.set_major_locator(mticker.MaxNLocator(integer=True))
        ax.legend(fontsize=7, framealpha=0.7)

    fig.tight_layout()
    path = os.path.join(plots_dir, 'summary.png')
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    return path


# ================================================================
# SAVE RESULTS JSON
# ================================================================

def save_results(out_dir, checkpoint_path, checkpoint_label,
                 domain_cfg, episode_data, env_info):
    os.makedirs(out_dir, exist_ok=True)
    rewards = [e['reward']  for e in episode_data]
    latency = [e['latency'] for e in episode_data]
    tx      = [e['tx']      for e in episode_data]
    noma    = [e['noma']    for e in episode_data]
    energy  = [e['energy']  for e in episode_data]

    summary = {
        'checkpoint'      : checkpoint_path,
        'checkpoint_label': checkpoint_label,
        'timestamp'       : datetime.now().isoformat(),
        'environment'     : domain_cfg,
        'active_uavs'     : env_info['num_uavs'],
        'n_episodes'      : len(episode_data),
        'results': {
            'reward' : {'mean': round(float(np.mean(rewards)), 4),
                        'std' : round(float(np.std(rewards)),  4)},
            'latency': {'mean': round(float(np.mean(latency)), 4),
                        'std' : round(float(np.std(latency)),  4)},
            'tx'     : {'mean': round(float(np.mean(tx)),      4),
                        'std' : round(float(np.std(tx)),       4)},
            'noma'   : {'mean': round(float(np.mean(noma)),    4),
                        'std' : round(float(np.std(noma)),     4)},
            'energy' : {'mean': round(float(np.mean(energy)),  6),
                        'std' : round(float(np.std(energy)),   6)},
        },
    }
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(out_dir, 'episodes.jsonl'), 'w') as f:
        for i, ep in enumerate(episode_data):
            f.write(json.dumps({'episode': i + 1, **ep}) + '\n')


# ================================================================
# MAIN TEST FUNCTION
# ================================================================

def test(args):
    device = torch.device(args.device)

    # ── Resolve checkpoint ────────────────────────────────────────
    if args.checkpoint:
        checkpoint_path  = args.checkpoint
        checkpoint_label = 'manually specified'
        if not os.path.exists(checkpoint_path):
            print(f"[ERROR] Checkpoint not found: {checkpoint_path}")
            sys.exit(1)
    else:
        checkpoint_path, checkpoint_label = find_best_checkpoint()
        if checkpoint_path is None:
            print("[ERROR] No checkpoint found in results/checkpoints/")
            print("        Train first, or use --checkpoint <path>")
            sys.exit(1)

    # ── Build domain_cfg ──────────────────────────────────────────
    domain_cfg = {
        'num_users'  : args.num_iot,
        'num_uavs'   : args.num_uavs,
        'num_jammers': args.num_jammers,
        'area_radius': args.area_radius,
    }
    if args.jammer_power_dbm is not None:
        domain_cfg['jammer_tx_power'] = _dbm_to_watt(args.jammer_power_dbm)
    if args.iot_power_dbm is not None:
        domain_cfg['iot_tx_power'] = _dbm_to_watt(args.iot_power_dbm)
    if args.wind_speed is not None:
        domain_cfg['wind_speed'] = args.wind_speed
    if args.jammer_radius is not None:
        domain_cfg['jammer_interference_radius'] = args.jammer_radius

    # ── Output directory ──────────────────────────────────────────
    ckpt_name = os.path.splitext(os.path.basename(checkpoint_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_dir   = os.path.join('results', 'test_results', f'{ckpt_name}_{timestamp}')
    log_dir   = os.path.join(out_dir, 'logs')
    traj_dir  = os.path.join(out_dir, 'trajectories')
    plots_dir = os.path.join(out_dir, 'plots')
    for d in [out_dir, log_dir, traj_dir, plots_dir]:
        os.makedirs(d, exist_ok=True)

    # ── Build env and agent ───────────────────────────────────────
    env   = MultiUAVMECEnv(seed=args.seed, domain_cfg=domain_cfg)
    agent = MAPPOAgent(
        obs_dim    = env.local_obs_dim,
        state_dim  = env.global_state_dim,
        action_dim = env.agent_action_dim,
        n_agents   = env.n_agents,
        device     = device,
        share_actor= True,
    )
    agent.load(checkpoint_path)

    env_info = {
        'num_iot'    : env.num_users,
        'num_uavs'   : env.num_uavs,
        'num_jammers': env.num_jammers,
        'area_radius': env.area_radius,
    }

    # ── Print header ──────────────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  Diffusion-MAPPO  —  Custom Environment Test")
    print(f"{'='*62}")
    print(f"  Checkpoint  : {os.path.basename(checkpoint_path)}")
    print(f"  Source      : {checkpoint_label}")
    print(f"  Device      : {device}")
    print(f"  Output dir  : {out_dir}")
    print(f"{'─'*62}")
    print(f"  IoT devices : {env.num_users}")
    print(f"  UAVs        : {env.num_uavs}  (active)  /  {env.n_agents} (padded)")
    print(f"  Jammers     : {env.num_jammers}")
    print(f"  Area radius : {env.area_radius} m")
    print(f"  Episodes    : {args.episodes}")
    print(f"{'='*62}\n")

    # ── Run episodes ──────────────────────────────────────────────
    episode_data = []
    step_log     = []        # per-step rows for training_metrics.jsonl
    global_step  = 0

    for ep in range(args.episodes):
        obs, state = env.reset(domain_cfg=domain_cfg)
        ep_reward  = 0.0
        ep_latency, ep_tx, ep_noma, ep_energy = [], [], [], []

        for _ in range(NUM_TIME_SLOTS):
            with torch.no_grad():
                actions, _, _ = agent.get_actions(
                    obs, state, num_active=env.num_active)
            obs, state, rewards, done, info = env.step(actions)
            global_step += 1

            r_mean = float(rewards.mean())
            ep_reward += r_mean
            ep_latency.append(info['avg_latency'])
            ep_tx.append(info['successful_tx'])
            ep_noma.append(info['noma_pairs_formed'])
            ep_energy.append(info['total_energy_uav'])

            # Accumulate per-step log (same schema as training logger)
            step_log.append({
                'step'          : global_step,
                'reward'        : round(r_mean, 5),
                'avg_latency'   : float(info.get('avg_latency',       0)),
                'total_energy'  : float(info.get('total_energy_uav',  0)),
                'successful_tx' : int(info.get('successful_tx',       0)),
                'noma_pairs'    : int(info.get('noma_pairs_formed',    0)),
                'local_compute' : int(info.get('local_compute',        0)),
                'uav_compute'   : int(info.get('uav_compute',          0)),
                'compute_fail'  : int(info.get('compute_fail',         0)),
                'cache_local'   : int(info.get('cache_local_hits',     0)),
                'cache_coop'    : int(info.get('cache_coop_hits',      0)),
                'cache_bs'      : int(info.get('cache_bs_fetches',     0)),
                'wind_speed'    : float(info.get('wind_speed',         0)),
                'wind_direction': float(info.get('wind_direction',     0)),
                'jammer_blocked': int(info.get('jammer_blocked',       0)),
                'num_active_uavs': int(info.get('num_active_uavs',    0)),
            })

            if done:
                break

        entry = {
            'reward' : round(ep_reward,                  4),
            'latency': round(float(np.mean(ep_latency)), 4),
            'tx'     : round(float(np.mean(ep_tx)),      4),
            'noma'   : round(float(np.mean(ep_noma)),    4),
            'energy' : round(float(np.mean(ep_energy)),  6),
        }
        episode_data.append(entry)

        print(f"  Episode {ep+1:3d}/{args.episodes} | "
              f"reward={entry['reward']:7.2f} | "
              f"latency={entry['latency']:6.1f} ms | "
              f"tx={entry['tx']:.1f} | "
              f"noma={entry['noma']:.1f} | "
              f"energy={entry['energy']:.4f} J")

    # ── Summary ───────────────────────────────────────────────────
    rewards = [e['reward']  for e in episode_data]
    latency = [e['latency'] for e in episode_data]
    tx      = [e['tx']      for e in episode_data]
    noma    = [e['noma']    for e in episode_data]
    energy  = [e['energy']  for e in episode_data]

    print(f"\n{'='*62}")
    print(f"  Results over {args.episodes} episodes")
    print(f"{'─'*62}")
    print(f"  Avg Reward       : {np.mean(rewards):8.2f}  ±  {np.std(rewards):.2f}")
    print(f"  Avg Latency      : {np.mean(latency):8.2f} ms")
    print(f"  Avg Successful TX: {np.mean(tx):8.2f}")
    print(f"  Avg NOMA Pairs   : {np.mean(noma):8.2f}")
    print(f"  Avg UAV Energy   : {np.mean(energy):8.4f} J")
    print(f"{'='*62}")

    # ── Save results JSON + JSONL ─────────────────────────────────
    save_results(out_dir, checkpoint_path, checkpoint_label,
                 domain_cfg, episode_data, env_info)
    print(f"\n  [Saved] metrics.json     → {out_dir}/")
    print(f"  [Saved] episodes.jsonl   → {out_dir}/")

    # ── Write per-step metrics for P1/P5 ─────────────────────────
    write_test_metrics(step_log, log_dir)
    print(f"  [Saved] training_metrics.jsonl → {log_dir}/")

    # ── Capture trajectory with trained agent ─────────────────────
    print(f"\n  Capturing trajectory with trained agent ...")
    npz_path = capture_trajectory(env, agent, traj_dir, step_tag=1)
    print(f"  [Saved] {os.path.basename(npz_path)} → {traj_dir}/")

    # ── Run parameter sweeps ──────────────────────────────────────
    print(f"\n  Running parameter sweeps (P2-P11 data) ...")
    print(f"  (this may take a few minutes)")
    try:
        sweeps_path = run_sweeps(env, agent, log_dir)
        print(f"  [Saved] eval_sweeps.json → {log_dir}/")
    except Exception as e:
        print(f"  [Warning] Sweeps failed: {e}  — P2-P4/P6-P7/P9-P11 will be skipped")

    # ── Generate all IEEE plots ───────────────────────────────────
    print(f"\n  Generating all plots ...")
    plots_dir_out, generated, skipped = generate_all_plots(out_dir, log_dir, traj_dir)
    print(f"\n  Summary plot ...")
    summary_path = generate_summary_plot(episode_data, plots_dir_out, env_info, ckpt_name)
    print(f"  [Saved] summary.png → {plots_dir_out}/")

    # ── Final output summary ──────────────────────────────────────
    print(f"\n{'='*62}")
    print(f"  All outputs saved to:  {out_dir}/")
    print(f"{'─'*62}")
    print(f"  {len(generated) + 1} plots generated  |  {len(skipped)} skipped")
    if skipped:
        print(f"  Skipped: {', '.join(skipped)}")
    print(f"{'='*62}\n")


# ================================================================
# HELPERS
# ================================================================

def _dbm_to_watt(dbm):
    return 10 ** ((dbm - 30) / 10)


# ================================================================
# ARGUMENT PARSER
# ================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test trained Diffusion-MAPPO on a custom environment',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # ── Checkpoint ─────────────────────────────────────────────────
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help='Path to a specific .pt checkpoint. If omitted, auto-selects: '
             'best_model.pt → final_model.pt → highest model_step*.pt')

    # ── Environment parameters ─────────────────────────────────────
    parser.add_argument('--num-iot',     type=int,   default=NUM_USERS,
                        help=f'Number of IoT devices '
                             f'(training default: {NUM_USERS}, range: 80-149)')
    parser.add_argument('--num-uavs',    type=int,   default=NUM_UAVS,
                        help=f'Number of UAVs '
                             f'(must be {MIN_UAVS}-{MAX_UAVS}, '
                             f'training default: {NUM_UAVS})')
    parser.add_argument('--num-jammers', type=int,   default=NUM_JAMMERS,
                        help=f'Number of jammers '
                             f'(training default: {NUM_JAMMERS}, range: 1-3)')
    parser.add_argument('--area-radius', type=float, default=AREA_RADIUS,
                        help=f'Coverage area radius in metres '
                             f'(training default: {AREA_RADIUS}, range: 400-600)')

    # ── Optional advanced overrides ────────────────────────────────
    parser.add_argument('--jammer-power-dbm', type=float, default=None,
                        help='Jammer TX power in dBm  (training range: 15-25)')
    parser.add_argument('--iot-power-dbm',    type=float, default=None,
                        help='IoT TX power in dBm    (training range: 15-23)')
    parser.add_argument('--wind-speed',       type=float, default=None,
                        help='Wind speed in m/s      (training range: 0.0-5.0)')
    parser.add_argument('--jammer-radius',    type=float, default=None,
                        help='Jammer interference radius in metres (range: 80-160)')

    # ── Test settings ───────────────────────────────────────────────
    parser.add_argument('--episodes', type=int,   default=10,
                        help='Number of test episodes to run')
    parser.add_argument('--seed',     type=int,   default=0,
                        help='Random seed for the environment')
    parser.add_argument('--device',   type=str,   default='cpu',
                        help='Compute device: cpu or cuda')

    args = parser.parse_args()

    if not (MIN_UAVS <= args.num_uavs <= MAX_UAVS):
        print(f"[ERROR] --num-uavs must be between {MIN_UAVS} and {MAX_UAVS}. "
              f"Got {args.num_uavs}.")
        sys.exit(1)

    test(args)