"""
Training Logger — saves all metrics to disk during training.

Files produced (all in results/logs/):
  training_metrics.jsonl  — one JSON line per step (reward, latency, energy, ...)
  eval_sweeps.json        — parameter sweep results (run at end of training)
  trajectory_XXXXXXX.npz  — UAV + IoT + jammer positions at step X (every 50k)
  cache_stats.jsonl       — per-step cache hit/miss counts
  episode_summary.jsonl   — per-episode totals

plot.py reads ONLY these files — no env simulation needed at plot time.

FIX (Bug 2): dev_cluster, dev_jammed, dev_pos shown in plots now use
END-of-episode positions so IoT colours match the final UAV (+) positions.
Previously dev_cluster_start was saved, causing a mismatch between the
UAV final position marker and the cluster colour of each IoT device.
"""

import os, json, time
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
LOG_DIR     = os.path.join(RESULTS_DIR, 'logs')
PLOT_DIR    = os.path.join(RESULTS_DIR, 'plots')
TRAJ_DIR    = os.path.join(RESULTS_DIR, 'trajectories')

for d in [LOG_DIR, PLOT_DIR, TRAJ_DIR]:
    os.makedirs(d, exist_ok=True)


class TrainingLogger:
    """
    Call log_step() every env.step() during training.
    Call save_trajectory() every 50k steps.
    Call run_and_save_sweeps() once at the end of training.
    """

    def __init__(self):
        self.metrics_path = os.path.join(LOG_DIR, 'training_metrics.jsonl')
        self.episode_path = os.path.join(LOG_DIR, 'episode_summary.jsonl')
        self.cache_path   = os.path.join(LOG_DIR, 'cache_stats.jsonl')
        self.sweep_path   = os.path.join(LOG_DIR, 'eval_sweeps.json')

        # Clear old logs
        for p in [self.metrics_path, self.episode_path, self.cache_path]:
            open(p, 'w').close()

        # In-memory episode buffer
        self._ep_rewards   = []
        self._ep_latencies = []
        self._ep_energies  = []
        self._ep_tx        = 0
        self._ep_noma      = 0

    def log_step(self, global_step, info, rewards_mean):
        """Call every env.step(). Appends one line to training_metrics.jsonl."""
        row = {
            'step':           global_step,
            'reward':         float(rewards_mean),
            'avg_latency':    float(info.get('avg_latency', 0)),
            'total_energy':   float(info.get('total_energy_uav', 0)),
            'successful_tx':  int(info.get('successful_tx', 0)),
            'noma_pairs':     int(info.get('noma_pairs_formed', 0)),
            'local_compute':  int(info.get('local_compute', 0)),
            'uav_compute':    int(info.get('uav_compute', 0)),
            'compute_fail':   int(info.get('compute_fail', 0)),
            'cache_local':    int(info.get('cache_local_hits', 0)),
            'cache_coop':     int(info.get('cache_coop_hits', 0)),
            'cache_bs':       int(info.get('cache_bs_fetches', 0)),
            'wind_speed':      float(info.get('wind_speed', 0)),
            'wind_direction':  float(info.get('wind_direction', 0)),
            'jammer_blocked': int(info.get('jammer_blocked', 0)),
            'num_active_uavs': int(info.get('num_active_uavs', 0)),
        }
        with open(self.metrics_path, 'a') as f:
            f.write(json.dumps(row) + '\n')

        # Episode accumulation
        self._ep_rewards.append(float(rewards_mean))
        self._ep_latencies.append(float(info.get('avg_latency', 0)))
        self._ep_energies.append(float(info.get('total_energy_uav', 0)))
        self._ep_tx   += int(info.get('successful_tx', 0))
        self._ep_noma += int(info.get('noma_pairs_formed', 0))

    def log_episode_end(self, global_step, episode_num):
        """Call when done=True. Writes episode summary."""
        if not self._ep_rewards:
            return
        row = {
            'step':         global_step,
            'episode':      episode_num,
            'mean_reward':  float(np.mean(self._ep_rewards)),
            'mean_latency': float(np.mean(self._ep_latencies)),
            'total_energy': float(np.sum(self._ep_energies)),
            'total_tx':     self._ep_tx,
            'total_noma':   self._ep_noma,
        }
        with open(self.episode_path, 'a') as f:
            f.write(json.dumps(row) + '\n')
        # Reset episode buffer
        self._ep_rewards   = []
        self._ep_latencies = []
        self._ep_energies  = []
        self._ep_tx   = 0
        self._ep_noma = 0

    def save_trajectory(self, global_step, env):
        """
        Save UAV + IoT + jammer positions at this step.
        Called every 50k steps during training.
        Produces: results/trajectories/trajectory_0050000.npz
        """
        uav_pos  = np.array([u.position.copy() for u in env.uavs[:env.num_active]])
        uav_ids  = np.arange(env.num_active)
        uav_caps = np.array([u.get_capability_vector() for u in env.uavs[:env.num_active]])

        dev_pos     = np.array([d.position.copy() for d in env.devices])
        dev_latency = np.array([d.latency for d in env.devices])
        dev_energy  = np.array([d.energy  for d in env.devices])
        dev_cluster = np.array([d.cluster_id for d in env.devices])
        dev_jammed  = np.array([env.is_in_jammer_zone(d) for d in env.devices])

        jam_pos    = (np.array([j.position.copy() for j in env.jammers])
                      if env.jammers else np.zeros((0, 3)))
        jam_radius = env.jammer_interference_radius

        path = os.path.join(TRAJ_DIR, f'trajectory_{global_step:07d}.npz')
        np.savez_compressed(path,
            step=global_step,
            area_radius=env.area_radius,
            num_active=env.num_active,
            uav_pos=uav_pos, uav_ids=uav_ids, uav_caps=uav_caps,
            dev_pos=dev_pos, dev_latency=dev_latency,
            dev_energy=dev_energy, dev_cluster=dev_cluster,
            dev_jammed=dev_jammed,
            jam_pos=jam_pos, jam_radius=jam_radius,
            wind_speed=env.wind_speed, wind_direction=env.wind_direction,
        )
        return path

    def save_trajectory_episode(self, global_step, env, n_steps=500, agent=None):
        """
        Run a short episode and save the FULL trajectory (all steps).
        Also captures real NOMA pair and cooperative offload snapshots
        from env.step() info dict for accurate plot rendering.

        FIX (Bug 1): creates a temporary environment with the same config
        rather than calling env.reset() on the TRAINING environment.
        The original code reset and ran steps on the live training env,
        corrupting obs/state in the rollout collection loop.

        FIX (Bug 2): dev_cluster and dev_jammed are now computed from
        END-of-episode positions (final UAV + IoT positions) instead of
        START positions. This ensures IoT dot colours in P8 match the
        UAV final position (+) markers correctly.

        FIX (Bug 3): when agent is supplied, the trained policy is used
        for action selection instead of a correlated random walk.  This
        produces realistic, space-filling trajectories that match actual
        learned behaviour and fills the full deployment area.
        """
        from config import MAX_UAVS

        # ── Create a throw-away env — never touch the training env ──────
        tmp_env = env.__class__(
            num_users   = env.num_users,
            num_uavs    = env.num_uavs,
            num_jammers = env.num_jammers,
            seed        = global_step,        # unique seed per snapshot
            domain_cfg  = env.domain_cfg,
        )
        obs_arr, state_arr = tmp_env.reset()

        # ── START snapshot — kept only for reference (uav_pos_start) ────
        # dev_pos_start: IoT positions at episode start (for trail drawing)
        dev_positions_start = np.array([d.position.copy() for d in tmp_env.devices])
        uav_pos_start       = np.array([u.position.copy()
                                        for u in tmp_env.uavs[:tmp_env.num_active]])

        # ── Trajectory containers ────────────────────────────────────────
        uav_trajectories = {i: [] for i in range(tmp_env.num_active)}

        # Record every device's position each slot for IoT trail drawing
        iot_trajectories = {i: [tmp_env.devices[i].position[:2].copy()]
                            for i in range(len(tmp_env.devices))}

        # Capture NOMA and coop offload snapshots during episode
        noma_snapshot = None   # (uav_id, uav_pos, [dev_pos_far, dev_pos_near])
        coop_snapshot = None   # (src_id, dst_id, src_pos, dst_pos)

        # ── Action source ────────────────────────────────────────────────
        # Prefer the trained policy for realistic trajectories.  Fall back
        # to a high-persistence correlated random walk (ρ=0.85) only when
        # no agent is supplied (e.g. early checkpointing before training).
        # High ρ keeps the direction consistent long enough for UAVs to
        # travel across the deployment area and produce space-filling paths.
        use_policy = agent is not None
        _rng_traj = np.random.RandomState(global_step % (2**31 - 1))
        _act = _rng_traj.uniform(-0.8, 0.8, (MAX_UAVS, 4))

        # obs_arr / state_arr already set by the reset() call above

        # ── Run episode ──────────────────────────────────────────────────
        for t in range(n_steps):
            if use_policy:
                import torch
                with torch.inference_mode():
                    actions, _, _ = agent.get_actions(
                        obs_arr, state_arr, num_active=tmp_env.num_active)
            else:
                # High-persistence correlated random walk (ρ=0.85)
                # Large ρ keeps direction stable → UAVs traverse the full area
                _act = np.clip(
                    0.85 * _act + 0.15 * _rng_traj.uniform(-1.0, 1.0, (MAX_UAVS, 4)),
                    -1.0, 1.0)
                actions = _act.copy()

            obs_arr, state_arr, _, done, info = tmp_env.step(actions)

            for i in range(tmp_env.num_active):
                # Save full 3D position (x, y, altitude) for 3D trajectory plot
                uav_trajectories[i].append(tmp_env.uavs[i].position[:3].copy())

            # Record IoT positions this slot
            for i, dev in enumerate(tmp_env.devices):
                if i in iot_trajectories:
                    iot_trajectories[i].append(dev.position[:2].copy())

            # Capture first NOMA pair we see
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
                        # Far user = farther from UAV, near user = closer
                        if np.linalg.norm(d0 - uav_xy) < np.linalg.norm(d1 - uav_xy):
                            d0, d1 = d1, d0
                        noma_snapshot = (aid, uav_xy, np.array([d0, d1]))
                        break

            # Capture first cooperative offload we see
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

        # ── END snapshot — use final positions for cluster/jammed labels ──
        #
        # Previously: dev_cluster_start was saved (positions at slot 0).
        # This caused IoT colours to reflect the old UAV arrangement, not
        # the final UAV positions shown by the (+) markers.
        #
        # Fix: recompute cluster assignment from END positions so that
        # each IoT dot colour matches its nearest UAV at episode END.

        dev_positions_end = np.array([d.position.copy() for d in tmp_env.devices])

        # Final UAV positions (2-D x,y only)
        uav_finals = np.array([tmp_env.uavs[i].position[:2]
                                for i in range(tmp_env.num_active)])

        # ── FIX: dev_cluster from END positions ─────────────────────────
        if tmp_env.num_active > 0:
            dev_cluster_end = np.array([
                int(np.argmin([
                    np.linalg.norm(d.position[:2] - uav_finals[j])
                    for j in range(tmp_env.num_active)
                ]))
                for d in tmp_env.devices
            ])
        else:
            dev_cluster_end = np.zeros(len(tmp_env.devices), dtype=int)

        # ── FIX: dev_jammed from END positions ──────────────────────────
        dev_jammed_end = np.array([
            tmp_env.is_in_jammer_zone(d) for d in tmp_env.devices
        ])

        # ── Communication range and in-range flags (END positions) ──────
        COMM_RANGE   = 200.0
        dev_in_range = np.ones(len(tmp_env.devices), dtype=bool)
        if tmp_env.num_active > 0:
            for idx, d in enumerate(tmp_env.devices):
                cid = dev_cluster_end[idx]          # use END cluster
                if 0 <= cid < tmp_env.num_active:
                    dist = np.linalg.norm(
                        d.position[:2] - uav_finals[cid]
                    )
                    dev_in_range[idx] = bool(dist <= COMM_RANGE)

        # ── Build trajectory arrays ──────────────────────────────────────
        traj_arrays = {}
        for i, traj in uav_trajectories.items():
            arr = np.array(traj)
            # Ensure 3-column (x, y, z): pad with profile altitude if needed
            if arr.ndim == 2 and arr.shape[1] == 2:
                from config import UAV_PROFILES
                alt = UAV_PROFILES[i % len(UAV_PROFILES)]["altitude"]
                arr = np.column_stack([arr, np.full(len(arr), alt)])
            traj_arrays[f'uav_{i}'] = arr

        MAX_IOT_TRAJ = 500
        iot_items = [(i, traj) for i, traj in iot_trajectories.items()
                     if len(traj) >= 2]
        if len(iot_items) > MAX_IOT_TRAJ:
            stride    = len(iot_items) // MAX_IOT_TRAJ
            iot_items = iot_items[::stride]
        for i, traj in iot_items:
            traj_arrays[f'iot_{i}'] = np.array(traj)

        # Jammer positions
        jam_pos = (np.array([j.position.copy() for j in tmp_env.jammers])
                   if tmp_env.jammers else np.zeros((0, 3)))

        # ── NOMA snapshot ────────────────────────────────────────────────
        if noma_snapshot is not None:
            noma_uav_id  = np.array([noma_snapshot[0]], dtype=np.int32)
            noma_uav_pos = noma_snapshot[1].astype(np.float32)
            noma_dev_pos = noma_snapshot[2].astype(np.float32)
        else:
            noma_uav_id  = np.array([-1], dtype=np.int32)
            noma_uav_pos = np.zeros(2,    dtype=np.float32)
            noma_dev_pos = np.zeros((0, 2), dtype=np.float32)

        # ── Cooperative offload snapshot ─────────────────────────────────
        if coop_snapshot is not None:
            coop_src_id  = np.array([coop_snapshot[0]], dtype=np.int32)
            coop_dst_id  = np.array([coop_snapshot[1]], dtype=np.int32)
            coop_src_pos = coop_snapshot[2].astype(np.float32)
            coop_dst_pos = coop_snapshot[3].astype(np.float32)
        else:
            coop_src_id  = np.array([-1], dtype=np.int32)
            coop_dst_id  = np.array([-1], dtype=np.int32)
            coop_src_pos = np.zeros(2,    dtype=np.float32)
            coop_dst_pos = np.zeros(2,    dtype=np.float32)

        # ── Save .npz ────────────────────────────────────────────────────
        path = os.path.join(TRAJ_DIR, f'flight_path_{global_step:07d}.npz')
        np.savez_compressed(path,
            step        = global_step,
            area_radius = tmp_env.area_radius,
            num_active  = tmp_env.num_active,
            jam_pos     = jam_pos,
            jam_radius  = tmp_env.jammer_interference_radius,
            uav_comm_range = COMM_RANGE,

            # Positions
            uav_pos_start    = uav_pos_start,
            dev_pos_start    = dev_positions_start,   # IoT start pos (trail drawing)
            dev_pos_end      = dev_positions_end,     # IoT end pos   (reference)

            # ── FIXED: cluster + jammed + in_range all from END ──────────
            dev_cluster  = dev_cluster_end,
            dev_jammed   = dev_jammed_end,
            dev_in_range = dev_in_range,              # already END (unchanged)

            # Wind
            wind_speed     = np.float32(tmp_env.wind_speed),
            wind_direction = np.float32(tmp_env.wind_direction),

            # NOMA pair snapshot
            noma_uav_id  = noma_uav_id,
            noma_uav_pos = noma_uav_pos,
            noma_dev_pos = noma_dev_pos,

            # Cooperative offload snapshot
            coop_src_id  = coop_src_id,
            coop_dst_id  = coop_dst_id,
            coop_src_pos = coop_src_pos,
            coop_dst_pos = coop_dst_pos,

            **traj_arrays,
        )
        return path

    def run_and_save_sweeps(self, env_class, n_episodes=3, n_steps=100):
        """
        Run parameter sweeps ONCE at end of training. Results saved to
        eval_sweeps.json so plot.py can read them instantly.
        """
        from config import (MAX_UAVS, dbm_to_watt, IOT_TX_POWER_MIN,
                            IOT_TX_POWER_MAX, TASK_DATA_SIZE_RANGE)
        from src.env import DomainRandomiser
        import config as cfg_mod

        print('  Running parameter sweeps for plots...')
        sweeps = {}

        def _run(env_kwargs, n_ep, n_st):
            lats, engs = [], []
            for ep in range(n_ep):
                env = env_class(seed=9000 + ep, **env_kwargs)
                env.reset()
                ep_lat, ep_eng = [], []
                for _ in range(n_st):
                    actions = np.random.uniform(-1, 1, (MAX_UAVS, 4))
                    _, _, _, done, info = env.step(actions)
                    ep_lat.append(info['avg_latency'])
                    ep_eng.append(info['total_energy_uav'])
                    if done: break
                lats.append(float(np.mean(ep_lat)))
                engs.append(float(np.sum(ep_eng)))
            return {'mean': float(np.mean(lats)), 'std': float(np.std(lats)),
                    'energy_mean': float(np.mean(engs)), 'energy_std': float(np.std(engs))}

        # P2: Latency vs N (IoT devices)
        print('    P2: sweeping N...')
        sweeps['P2'] = {'x': [], 'y': [], 'yerr': [],
                        'xlabel': 'Number of IoT Devices ($N$)',
                        'ylabel': 'Average Latency (s)',
                        'title':  'Average Latency vs. Number of IoT Devices'}
        for N in [60, 80, 100, 120, 140, 160]:
            cfg = DomainRandomiser().default(); cfg['num_users'] = N
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P2']['x'].append(N)
            sweeps['P2']['y'].append(r['mean'])
            sweeps['P2']['yerr'].append(r['std'])

        # P3: Latency vs M (UAVs)
        print('    P3: sweeping M...')
        sweeps['P3'] = {'x': [], 'y': [], 'yerr': [],
                        'xlabel': 'Number of UAVs ($M$)',
                        'ylabel': 'Average Latency (s)',
                        'title':  'Average Latency vs. Number of UAVs'}
        for M in [3, 4, 5, 6, 7, 8]:
            cfg = DomainRandomiser().default(); cfg['num_uavs'] = M
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P3']['x'].append(M)
            sweeps['P3']['y'].append(r['mean'])
            sweeps['P3']['yerr'].append(r['std'])

        # P4: Latency vs J (jammers)
        print('    P4: sweeping J...')
        sweeps['P4'] = {'x': [], 'y': [], 'yerr': [],
                        'xlabel': 'Number of Jammers ($J$)',
                        'ylabel': 'Average Latency (s)',
                        'title':  'Latency vs. Number of Jammers'}
        for J in [0, 1, 2, 3, 4]:
            cfg = DomainRandomiser().default(); cfg['num_jammers'] = J
            r = _run({'num_jammers': J, 'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P4']['x'].append(J)
            sweeps['P4']['y'].append(r['mean'])
            sweeps['P4']['yerr'].append(r['std'])

        # P5: Energy — saved from training_metrics.jsonl (no sweep needed)
        sweeps['P5'] = {'note': 'Read from training_metrics.jsonl'}

        # P6: Latency vs cache capacity
        print('    P6: sweeping cache...')
        import copy
        sweeps['P6'] = {'x': [], 'y': [], 'yerr': [],
                        'xlabel': 'Average Cache Capacity (programs)',
                        'ylabel': 'Average Latency (s)',
                        'title':  'Latency vs. Cache Capacity'}
        from config import UAV_PROFILES
        for sf in [0.25, 0.5, 1.0, 1.5, 2.0, 3.0]:
            orig = copy.deepcopy(UAV_PROFILES)
            for p in UAV_PROFILES:
                p['cache_programs'] = max(2, int(p['cache_programs'] * sf))
            avg_cap = np.mean([p['cache_programs'] for p in UAV_PROFILES[:5]])
            cfg = DomainRandomiser().default()
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P6']['x'].append(float(avg_cap))
            sweeps['P6']['y'].append(r['mean'])
            sweeps['P6']['yerr'].append(r['std'])
            for i, p in enumerate(orig): UAV_PROFILES[i] = p

        # P7: Cache hit rate vs Zipf
        print('    P7: sweeping Zipf...')
        from src.caching import CooperativeCacheManager
        sweeps['P7'] = {'x': [], 'local': [], 'coop': [], 'bs': [],
                        'xlabel': 'Zipf Exponent',
                        'ylabel': 'Cache Hit Rate (%)',
                        'title':  'Cache Hit Rate vs. Zipf Exponent'}
        for z in [0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6]:
            cm = CooperativeCacheManager(
                num_uavs=5, per_uav_capacities=[14, 10, 6, 12, 8], seed=42)
            ranks = np.arange(1, cm.num_programs + 1, dtype=float)
            cm.program_popularity = (ranks ** (-z))
            cm.program_popularity /= cm.program_popularity.sum()
            rng = np.random.RandomState(42)
            loc, coo, bs = 0, 0, 0
            for _ in range(1000):
                tier, _, _ = cm.lookup(
                    rng.randint(0, 5),
                    rng.choice(cm.num_programs, p=cm.program_popularity))
                if tier == 1:   loc += 1
                elif tier == 2: coo += 1
                else:           bs  += 1
            t = loc + coo + bs
            sweeps['P7']['x'].append(z)
            sweeps['P7']['local'].append(loc / t * 100)
            sweeps['P7']['coop'].append(coo / t * 100)
            sweeps['P7']['bs'].append(bs  / t * 100)

        # P9: Latency vs jammer power
        print('    P9: sweeping jammer power...')
        sweeps['P9'] = {'x': [], 'y': [], 'yerr': [],
                        'xlabel': 'Jammer Transmit Power (dBm)',
                        'ylabel': 'Average Latency (s)',
                        'title':  'Latency vs. Jammer Power'}
        for p_dbm in [10, 15, 20, 25, 30]:
            cfg = DomainRandomiser().default()
            cfg['jammer_tx_power'] = dbm_to_watt(p_dbm)
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P9']['x'].append(p_dbm)
            sweeps['P9']['y'].append(r['mean'])
            sweeps['P9']['yerr'].append(r['std'])

        # P10: Latency vs task size
        print('    P10: sweeping task size...')
        sweeps['P10'] = {'x_labels': [], 'y': [], 'yerr': [],
                         'xlabel': 'Task Data Size Range (Mbits)',
                         'ylabel': 'Average Latency (s)',
                         'title':  'Latency vs. Task Size'}
        orig_range = cfg_mod.TASK_DATA_SIZE_RANGE
        for sz, label in [((0.2e6, 1e6),   '0.2-1'),
                          ((0.5e6, 2.5e6),  '0.5-2.5'),
                          ((0.5e6, 5e6),    '0.5-5'),
                          ((1e6,   8e6),    '1-8'),
                          ((2e6,   12e6),   '2-12')]:
            cfg_mod.TASK_DATA_SIZE_RANGE = sz
            cfg = DomainRandomiser().default()
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P10']['x_labels'].append(label)
            sweeps['P10']['y'].append(r['mean'])
            sweeps['P10']['yerr'].append(r['std'])
        cfg_mod.TASK_DATA_SIZE_RANGE = orig_range

        # P11: Latency vs transmission power
        print('    P11: sweeping TX power...')
        sweeps['P11'] = {'x': [], 'y': [], 'yerr': [],
                         'xlabel': 'IoT Transmission Power (dBm)',
                         'ylabel': 'Average Latency (s)',
                         'title':  'Latency vs. TX Power'}
        orig_min = cfg_mod.IOT_TX_POWER_MIN
        orig_max = cfg_mod.IOT_TX_POWER_MAX
        for p_dbm in [10, 15, 18, 20, 23, 25]:
            cfg_mod.IOT_TX_POWER_MIN = dbm_to_watt(p_dbm)
            cfg_mod.IOT_TX_POWER_MAX = dbm_to_watt(p_dbm)
            cfg = DomainRandomiser().default()
            r = _run({'domain_cfg': cfg}, n_episodes, n_steps)
            sweeps['P11']['x'].append(p_dbm)
            sweeps['P11']['y'].append(r['mean'])
            sweeps['P11']['yerr'].append(r['std'])
        cfg_mod.IOT_TX_POWER_MIN = orig_min
        cfg_mod.IOT_TX_POWER_MAX = orig_max

        with open(self.sweep_path, 'w') as f:
            json.dump(sweeps, f, indent=2)
        print(f'  Sweeps saved to {self.sweep_path}')