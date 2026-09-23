"""
Phase-2 UAV-MEC environment
============================
Extends Phase-1 with:
  - Explicit task lifecycle: ARRIVED → ASSIGNED → UPLOADING → QUEUED → COMPUTING → COMPLETED
  - Persistent FIFO queues; unfinished tasks span multiple slots
  - Wall-clock timing with strict identity: T_total = T_upload + T_queue + T_compute
  - Deadline tracking per task (deadline_missed flag)
  - Battery-depletion enforcement: depleted UAVs stop moving and reject new tasks
  - Episode-level percentile latency (p50 / p95 / p99)
  - Expanded info dict with tasks_pending, deadline_misses, energy aliases

Fixed IoT users --A2G--> UAV upload queue --> CPU queue --> completed task

Intentionally NOT in this phase:
  user mobility, NOMA/SIC, jammers, wind, A2A, service chains,
  caching/replication, two-timescale control, diffusion policies.

Public interface (MAPPO-compatible):
  reset() -> obs, state
  step(actions) -> next_obs, next_state, rewards, done, info
"""

from collections import deque

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import (
    AREA_RADIUS, NUM_USERS, MAX_NUM_USERS, NUM_UAVS, MAX_UAVS, MIN_UAVS,
    UAV_PROFILES, NUM_UAV_CAPABILITY_DIMS,
    IOT_TX_POWER, NOISE_PSD,
    TASK_DATA_SIZE_RANGE, TASK_CPU_CYCLES_RANGE, TASK_DEADLINE_RANGE,
    TASK_GENERATION_PROB,
    NUM_TIME_SLOTS, SLOT_DURATION,
    UAV_ALT_MIN, UAV_ALT_MAX, UAV_ALT_DELTA_MAX,
    ALPHA_LATENCY, BETA_ENERGY, EFFECTIVE_CAPACITANCE,
    PHASE1_OBS_USERS, PHASE1_UAV_BATTERY_J, PHASE1_LAT_NORM,
    PHASE1_ENERGY_NORM, PHASE1_UAV_RX_POWER_W, PHASE1_ACTION_DIM,
    DEFAULT_SERVICE_CHAIN, DEFAULT_SERVICE_PLACEMENT,
    dbm_to_watt,
)
from src.channel_model import ChannelModel
from src.association import associate_nearest
from src.latency import (
    calculate_a2g_distance,
    calculate_upload_latency,
    calculate_queue_latency,
    calculate_compute_latency,
    calculate_a2a_latency,
    calculate_total_latency,
)
from src.metrics import uav_propulsion_energy, computation_energy


# ================================================================
# ENTITIES
# ================================================================
class IoTDevice:
    """Stationary IoT user. Position is fixed for the whole episode."""

    def __init__(self, device_id, position, tx_power=IOT_TX_POWER):
        self.id = device_id
        self.position = np.array(position, dtype=float)
        self.tx_power = tx_power
        self.pending_tasks = deque()

    def reset(self):
        self.pending_tasks.clear()


class UAV:
    def __init__(self, uav_id, position, profile=None):
        self.id = uav_id
        self.position = np.array(position, dtype=float)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.cpu_freq = 2e9
        self.bandwidth = 10e6
        self.tx_power = 0.01
        self.max_displacement = 50.0
        self.battery_energy = PHASE1_UAV_BATTERY_J
        self.battery_max = PHASE1_UAV_BATTERY_J
        self.total_energy = 0.0
        self.task_queue = deque()
        self.upload_buffer = deque()
        self.active_upload = None
        self.computing = None
        if profile is not None:
            self.cpu_freq = profile["cpu_freq"]
            self.bandwidth = profile["bandwidth"]
            self.tx_power = dbm_to_watt(profile["tx_power_dbm"])
            self.max_displacement = profile["max_displacement"]
            self.position[2] = profile["altitude"]

    def reset(self, position=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.battery_energy = self.battery_max
        self.total_energy = 0.0
        self.task_queue.clear()
        self.upload_buffer.clear()
        self.active_upload = None
        self.computing = None

    @property
    def cpu_utilization(self):
        return 1.0 if self.computing is not None else 0.0

    def get_capability_vector(self):
        return np.array([
            self.cpu_freq / 3e9,
            0.0,
            self.bandwidth / 15e6,
            self.tx_power / dbm_to_watt(27),
            self.position[2] / 120.0,
        ], dtype=np.float32)


class ServiceStage:
    """
    Single stage in an ordered service chain (Phase 3).
    """

    def __init__(self, name, uav_id, cpu_cycles):
        self.name = str(name)
        self.uav_id = int(uav_id)
        self.cpu_cycles = float(cpu_cycles)
        self.remaining_cycles = float(cpu_cycles)
        self.enqueue_time = None
        self.compute_start_time = None
        self.compute_finish_time = None
        self.compute_elapsed = 0.0
        self.upload_buffer_wait = 0.0
        self.queue_wait = 0.0
        self.t_queue = 0.0
        self.t_compute = 0.0
        self.energy = 0.0
        self.status = "pending"  # pending -> queued -> computing -> completed


class Task:
    _next_id = 0

    def __init__(self, user_id, arrival_time, data_size_bits, cpu_cycles, deadline,
                 chain=None, placement=None):
        self.task_id = Task._next_id
        Task._next_id += 1
        self.user_id = user_id
        self.arrival_time = float(arrival_time)
        self.data_size_bits = float(data_size_bits)
        self.cpu_cycles = float(cpu_cycles)
        self.deadline = float(deadline)
        self.assigned_uav_id = None
        self.remaining_bits = float(data_size_bits)
        self.remaining_cycles = float(cpu_cycles)
        # Upload timing
        self.upload_start_time = None
        self.upload_finish_time = None    # Phase-2: wall-clock upload completion time
        self.enqueue_time = None
        # Compute timing
        self.compute_start_time = None
        self.compute_finish_time = None   # Phase-2: wall-clock compute completion time
        # Latency components (seconds)
        self.t_upload = 0.0
        self.t_queue = 0.0
        self.t_compute = 0.0
        self.t_a2a = 0.0
        self.t_total = 0.0
        # Channel snapshot at last active upload slot
        self.last_a2g_distance = 0.0
        self.last_a2g_rate = 0.0
        self.allocated_cpu_hz = 0.0
        # Elapsed counters (for multi-slot spanning)
        self.upload_elapsed = 0.0
        self.compute_elapsed = 0.0
        # Phase-2: deadline tracking
        self.deadline_missed = False      # True if t_total > deadline
        # Lifecycle status
        self.status = "generated"

        # Phase-3: Service Chain
        c = list(DEFAULT_SERVICE_CHAIN if chain is None else chain)
        p = dict(DEFAULT_SERVICE_PLACEMENT if placement is None else placement)
        self.chain = c
        self.current_stage_idx = 0
        n_stages = max(len(c), 1)
        stage_cyc = self.cpu_cycles / n_stages
        self.stages = [
            ServiceStage(
                name=s_name,
                uav_id=p.get(s_name, idx % MAX_UAVS),
                cpu_cycles=stage_cyc,
            )
            for idx, s_name in enumerate(self.chain)
        ]

    @property
    def current_stage(self):
        if 0 <= self.current_stage_idx < len(self.stages):
            return self.stages[self.current_stage_idx]
        return None

    @classmethod
    def reset_ids(cls):
        cls._next_id = 0


class DomainRandomiser:
    """
    Kept so existing training scripts can still import it.
    Phase 1/2 does not apply jammer/wind randomisation.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState()

    def default(self):
        return {
            "num_users": NUM_USERS,
            "num_uavs": NUM_UAVS,
            "area_radius": AREA_RADIUS,
        }

    def sample(self):
        return self.default()


# ================================================================
# ENVIRONMENT
# ================================================================
class MultiUAVMECEnv:
    """
    Phase-2 CTDE environment.

    Action per agent (length 5, values in [-1, 1]):
      a[0] dx  — x velocity command (× max_displacement metres / slot)
      a[1] dy  — y velocity command
      a[2] dh  — altitude change (× UAV_ALT_DELTA_MAX)
      a[3], a[4] unused padding (MAPPO width compatibility)

    Scheduling is heuristic: nearest-UAV association + FIFO compute.

    Phase-2 additions (vs Phase-1):
      - Persistent queues: tasks span multiple slots for upload and compute.
      - Explicit timing: t_queue = upload_buffer_wait + cpu_queue_wait.
      - Deadline tracking per completed task.
      - Battery-depletion: UAV stops moving and rejects new tasks.
      - Episode-level p50/p95/p99 latency in info dict.
    """

    def __init__(self, num_users=NUM_USERS, num_uavs=NUM_UAVS,
                 num_jammers=0, seed=42, domain_cfg=None,
                 task_gen_prob=None, service_chain=None,
                 service_placement=None):
        self.rng = np.random.RandomState(seed)
        self.base_seed = seed
        merged = {
            "num_users": num_users,
            "num_uavs": num_uavs,
            "area_radius": AREA_RADIUS,
        }
        if domain_cfg:
            merged.update(domain_cfg)
        self.domain_cfg = merged
        self.num_users = int(self.domain_cfg.get("num_users", num_users))
        self.num_uavs = int(np.clip(
            self.domain_cfg.get("num_uavs", num_uavs), MIN_UAVS, MAX_UAVS))
        self.n_agents = MAX_UAVS
        self.num_active = self.num_uavs
        self.area_radius = float(self.domain_cfg.get("area_radius", AREA_RADIUS))
        self.task_gen_prob = (TASK_GENERATION_PROB if task_gen_prob is None
                              else float(task_gen_prob))
        self.service_chain = list(
            DEFAULT_SERVICE_CHAIN if service_chain is None else service_chain)
        self.service_placement = dict(
            DEFAULT_SERVICE_PLACEMENT if service_placement is None else service_placement)

        # Stubs so leftover DMJO tooling does not crash on attribute access.
        self.num_jammers = 0
        self.jammers = []
        self.wind_speed = 0.0
        self.wind_direction = 0.0

        self.channel = ChannelModel()
        self.channel.wind_speed = 0.0
        self.t = 0
        self._init_entities()

        self.max_cluster_size = PHASE1_OBS_USERS
        self.local_obs_dim = self._compute_local_obs_dim()
        self.global_state_dim = self._compute_global_state_dim()
        self.agent_action_dim = self._compute_agent_action_dim()
        self.state_dim = self.global_state_dim
        self.action_dim = self.agent_action_dim * self.n_agents

        # Slot-level tracking
        self.completed_tasks = []
        self._slot_generated = 0
        self._slot_completed = []
        self._slot_deadline_misses = 0

        # Phase-2: episode-level accumulators for percentile latency and stats
        self._episode_latencies: list = []
        self._episode_tasks_generated = 0
        self._episode_tasks_completed = 0
        self._episode_deadline_misses = 0

    # ----------------------------------------------------------
    # Construction
    # ----------------------------------------------------------
    def _init_entities(self):
        self.devices = []
        for i in range(self.num_users):
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.area_radius * np.sqrt(self.rng.uniform())
            pos = (r * np.cos(angle), r * np.sin(angle), 0.0)
            self.devices.append(IoTDevice(i, pos, tx_power=IOT_TX_POWER))

        self.uavs = []
        for i in range(self.num_uavs):
            profile = UAV_PROFILES[i % len(UAV_PROFILES)]
            angle = 2 * np.pi * i / max(self.num_uavs, 1)
            r = self.area_radius * 0.5
            pos = (r * np.cos(angle), r * np.sin(angle), profile["altitude"])
            self.uavs.append(UAV(i, pos, profile=profile))

    def _sim_time(self):
        return self.t * SLOT_DURATION

    def _active_uavs(self):
        return self.uavs[:self.num_active]

    # ----------------------------------------------------------
    # Dimensions
    # ----------------------------------------------------------
    def _compute_local_obs_dim(self):
        """
        own: x, y, alt, vx, vy, battery, cpu_util          (7)
        active mask                                        (1)
        other UAV relative xy                              (2*(MAX_UAVS-1))
        other UAV active masks                             (MAX_UAVS-1)
        nearest users × (rel_x, rel_y, pending, size, cyc, deadline)
        queue length                                       (1)
        time fraction                                      (1)
        """
        k = PHASE1_OBS_USERS
        return (7 + 1
                + 2 * (MAX_UAVS - 1) + (MAX_UAVS - 1)
                + k * 6 + 1 + 1)

    def _compute_global_state_dim(self):
        """
        num_active (1)
        per UAV padded: xy (2) + capability (5) + battery (1) + cpu (1) + mask (1)
        per user padded: xy (2) + pending (1)
        time (1)
        """
        per_uav = 2 + NUM_UAV_CAPABILITY_DIMS + 1 + 1 + 1
        per_user = 3
        return 1 + MAX_UAVS * per_uav + MAX_NUM_USERS * per_user + 1

    def _compute_agent_action_dim(self):
        return PHASE1_ACTION_DIM

    # ----------------------------------------------------------
    # Physics helpers (public for tests)
    # ----------------------------------------------------------
    def calculate_a2g_distance(self, user_pos, uav_pos):
        return calculate_a2g_distance(user_pos, uav_pos)

    def calculate_a2g_rate(self, user, uav):
        return self.channel.a2g_uplink_rate(
            user.position, uav.position, user.tx_power, uav.bandwidth, NOISE_PSD)

    def calculate_upload_latency(self, data_size_bits, rate_bps):
        return calculate_upload_latency(data_size_bits, rate_bps)

    def calculate_queue_latency(self, enqueue_time, compute_start_time):
        return calculate_queue_latency(enqueue_time, compute_start_time)

    def calculate_compute_latency(self, cpu_cycles, allocated_cpu_hz):
        return calculate_compute_latency(cpu_cycles, allocated_cpu_hz)

    def calculate_a2a_latency(self, *args, **kwargs):
        return calculate_a2a_latency(*args, **kwargs)

    def calculate_total_latency(self, t_upload, t_queue, t_compute, t_a2a=0.0):
        return calculate_total_latency(t_upload, t_queue, t_compute, t_a2a)

    def associate_task(self, task):
        """
        Modular assignment hook. Phase 1/2: nearest UAV.
        Phase-2: battery-depleted UAVs cannot accept new tasks.
        Returns the assigned uav_id, or None if all UAVs are depleted.
        """
        user = self.devices[task.user_id]
        # Filter to active UAVs with remaining battery
        available = [u for u in self._active_uavs() if u.battery_energy > 0.0]
        if not available:
            return None  # all active UAVs depleted; caller should drop the task
        uav_id, dist = associate_nearest(user.position, available)
        task.assigned_uav_id = uav_id
        task.last_a2g_distance = dist
        return uav_id

    # ----------------------------------------------------------
    # Observations
    # ----------------------------------------------------------
    def get_local_obs(self, agent_id):
        r = self.area_radius
        obs = []
        is_active = 1.0 if agent_id < self.num_active else 0.0
        alt_range = max(UAV_ALT_MAX - UAV_ALT_MIN, 1.0)

        if agent_id < self.num_active:
            uav = self.uavs[agent_id]
            max_speed = max(uav.max_displacement, 1e-6)
            obs.extend([
                uav.position[0] / r,
                uav.position[1] / r,
                (uav.position[2] - UAV_ALT_MIN) / alt_range,
                uav.vx / max_speed,
                uav.vy / max_speed,
                uav.battery_energy / max(uav.battery_max, 1e-6),
                uav.cpu_utilization,
            ])
        else:
            obs.extend([0.0] * 7)
        obs.append(is_active)

        for j in range(MAX_UAVS):
            if j == agent_id:
                continue
            if j < self.num_active and agent_id < self.num_active:
                other = self.uavs[j]
                uav = self.uavs[agent_id]
                obs.extend([
                    (other.position[0] - uav.position[0]) / r,
                    (other.position[1] - uav.position[1]) / r,
                ])
            else:
                obs.extend([0.0, 0.0])

        for j in range(MAX_UAVS):
            if j == agent_id:
                continue
            obs.append(1.0 if j < self.num_active else 0.0)

        # Nearest users (or first K if inactive)
        if agent_id < self.num_active:
            uav = self.uavs[agent_id]
            order = np.argsort([
                np.linalg.norm(d.position[:2] - uav.position[:2])
                for d in self.devices
            ])
        else:
            order = np.arange(len(self.devices))

        size_max = TASK_DATA_SIZE_RANGE[1]
        cyc_max = TASK_CPU_CYCLES_RANGE[1]
        dl_max = TASK_DEADLINE_RANGE[1]
        k = PHASE1_OBS_USERS
        for n in range(k):
            if n < len(order) and agent_id < self.num_active:
                dev = self.devices[int(order[n])]
                uav = self.uavs[agent_id]
                pending = 1.0 if len(dev.pending_tasks) > 0 else 0.0
                if dev.pending_tasks:
                    task = dev.pending_tasks[0]
                    sz = task.data_size_bits / size_max
                    cyc = task.cpu_cycles / cyc_max
                    dl = task.deadline / dl_max
                else:
                    sz = cyc = dl = 0.0
                obs.extend([
                    (dev.position[0] - uav.position[0]) / r,
                    (dev.position[1] - uav.position[1]) / r,
                    pending, sz, cyc, dl,
                ])
            else:
                obs.extend([0.0] * 6)

        if agent_id < self.num_active:
            qlen = (len(self.uavs[agent_id].task_queue)
                    + (1 if self.uavs[agent_id].computing else 0)
                    + (1 if self.uavs[agent_id].active_upload else 0)
                    + len(self.uavs[agent_id].upload_buffer))
            obs.append(min(qlen / 20.0, 1.0))
        else:
            obs.append(0.0)
        obs.append(self.t / max(NUM_TIME_SLOTS, 1))
        return np.array(obs, dtype=np.float32)

    def get_all_local_obs(self):
        return np.stack([self.get_local_obs(i) for i in range(MAX_UAVS)])

    def get_global_state(self):
        r = self.area_radius
        state = [self.num_active / MAX_UAVS]
        for i in range(MAX_UAVS):
            if i < self.num_active:
                uav = self.uavs[i]
                state.extend([uav.position[0] / r, uav.position[1] / r])
                state.extend(uav.get_capability_vector().tolist())
                state.append(uav.battery_energy / max(uav.battery_max, 1e-6))
                state.append(uav.cpu_utilization)
                state.append(1.0)
            else:
                state.extend([0.0] * (2 + NUM_UAV_CAPABILITY_DIMS + 1 + 1 + 1))
        for i in range(MAX_NUM_USERS):
            if i < self.num_users:
                d = self.devices[i]
                state.extend([
                    d.position[0] / r,
                    d.position[1] / r,
                    1.0 if d.pending_tasks else 0.0,
                ])
            else:
                state.extend([0.0, 0.0, 0.0])
        state.append(self.t / max(NUM_TIME_SLOTS, 1))
        return np.array(state, dtype=np.float32)

    def get_state(self):
        return self.get_global_state()

    # ----------------------------------------------------------
    # Reset / step
    # ----------------------------------------------------------
    def reset(self, domain_cfg=None):
        self.t = 0
        Task.reset_ids()
        if domain_cfg is not None:
            self.domain_cfg = domain_cfg
            self.num_users = int(domain_cfg.get("num_users", self.num_users))
            self.num_uavs = int(np.clip(
                domain_cfg.get("num_uavs", self.num_uavs), MIN_UAVS, MAX_UAVS))
            self.num_active = self.num_uavs
            self.area_radius = float(domain_cfg.get("area_radius", self.area_radius))
        self.jammers = []
        self.wind_speed = 0.0
        self.channel.wind_speed = 0.0
        self._init_entities()
        # Slot-level trackers
        self.completed_tasks = []
        self._slot_generated = 0
        self._slot_completed = []
        self._slot_deadline_misses = 0
        # Phase-2: reset episode accumulators
        self._episode_latencies = []
        self._episode_tasks_generated = 0
        self._episode_tasks_completed = 0
        self._episode_deadline_misses = 0
        return self.get_all_local_obs(), self.get_global_state()

    def step(self, actions):
        actions = np.asarray(actions, dtype=float)
        if actions.ndim == 1:
            actions = actions.reshape(self.n_agents, self.agent_action_dim)
        self.t += 1
        now = self._sim_time()
        # Reset slot-level trackers
        self._slot_generated = 0
        self._slot_completed = []
        self._slot_deadline_misses = 0

        e_flight = np.zeros(MAX_UAVS)
        e_comm = np.zeros(MAX_UAVS)
        e_comp = np.zeros(MAX_UAVS)

        self._move_uavs(actions, e_flight)
        self._generate_tasks(now)
        self._process_uploads(now, e_comm)
        self._process_compute(now, e_comp)

        e_total_vec = e_flight + e_comm + e_comp
        for i, uav in enumerate(self._active_uavs()):
            spent = float(e_total_vec[i])
            uav.total_energy += spent
            uav.battery_energy = max(0.0, uav.battery_energy - spent)

        # Phase-2: update episode accumulators after processing
        self._episode_tasks_generated += self._slot_generated
        self._episode_tasks_completed += len(self._slot_completed)
        self._episode_deadline_misses += self._slot_deadline_misses
        for t in self._slot_completed:
            self._episode_latencies.append(t.t_total)

        rewards, info = self._pack_reward_info(e_flight, e_comm, e_comp, e_total_vec)
        done = self.t >= NUM_TIME_SLOTS
        return self.get_all_local_obs(), self.get_global_state(), rewards, done, info

    # ----------------------------------------------------------
    # Step internals
    # ----------------------------------------------------------
    def _move_uavs(self, actions, e_flight):
        r_limit = self.area_radius * 0.88
        for agent_id, uav in enumerate(self._active_uavs()):
            # Phase-2: battery-depleted UAVs cannot move
            if uav.battery_energy <= 0.0:
                uav.vx = 0.0
                uav.vy = 0.0
                uav.vz = 0.0
                e_flight[agent_id] = 0.0
                continue

            a = actions[agent_id]
            max_speed = uav.max_displacement
            uav.vx = float(np.clip(a[0], -1.0, 1.0)) * max_speed
            uav.vy = float(np.clip(a[1], -1.0, 1.0)) * max_speed
            dh = float(np.clip(a[2], -1.0, 1.0)) * UAV_ALT_DELTA_MAX
            new_h = float(np.clip(uav.position[2] + dh, UAV_ALT_MIN, UAV_ALT_MAX))
            uav.vz = new_h - uav.position[2]
            uav.position[2] = new_h

            target = uav.position[:2] + np.array([uav.vx, uav.vy])
            dist = np.linalg.norm(target)
            if dist > r_limit:
                target = target * (r_limit / dist)
                uav.vx = target[0] - uav.position[0]
                uav.vy = target[1] - uav.position[1]
            uav.position[0] = target[0]
            uav.position[1] = target[1]

            speed = float(np.hypot(uav.vx, uav.vy)) / max(SLOT_DURATION, 1e-9)
            e_flight[agent_id] = uav_propulsion_energy(speed, SLOT_DURATION)

    def _generate_tasks(self, now):
        n = len(self.devices)
        probs = self.rng.rand(n)
        cycles = self.rng.uniform(TASK_CPU_CYCLES_RANGE[0], TASK_CPU_CYCLES_RANGE[1], n)
        sizes = self.rng.uniform(TASK_DATA_SIZE_RANGE[0], TASK_DATA_SIZE_RANGE[1], n)
        deadlines = self.rng.uniform(TASK_DEADLINE_RANGE[0], TASK_DEADLINE_RANGE[1], n)
        for i, dev in enumerate(self.devices):
            if probs[i] >= self.task_gen_prob:
                continue
            task = Task(
                user_id=dev.id,
                arrival_time=now,
                data_size_bits=float(sizes[i]),
                cpu_cycles=float(cycles[i]),
                deadline=float(deadlines[i]),
                chain=self.service_chain,
                placement=self.service_placement,
            )
            # Phase-2: associate_task returns None if all UAVs are depleted
            assigned = self.associate_task(task)
            if assigned is None:
                continue  # drop task; no available UAV
            uav = self.uavs[task.assigned_uav_id]
            task.status = "assigned"
            uav.upload_buffer.append(task)
            dev.pending_tasks.append(task)
            self._slot_generated += 1

    def _process_uploads(self, now, e_comm):
        for uav in self._active_uavs():
            if uav.active_upload is None and uav.upload_buffer:
                task = uav.upload_buffer.popleft()
                task.status = "uploading"
                task.upload_start_time = now
                uav.active_upload = task

            task = uav.active_upload
            if task is None:
                continue
            user = self.devices[task.user_id]
            rate = self.calculate_a2g_rate(user, uav)
            dist = self.calculate_a2g_distance(user.position, uav.position)
            task.last_a2g_rate = rate
            task.last_a2g_distance = dist
            bits = rate * SLOT_DURATION
            task.remaining_bits = max(0.0, task.remaining_bits - bits)
            task.upload_elapsed += SLOT_DURATION
            e_comm[uav.id] += PHASE1_UAV_RX_POWER_W * SLOT_DURATION

            if task.remaining_bits <= 1e-9:
                # Phase-2: record wall-clock upload finish time
                task.upload_finish_time = now
                # t_upload: theoretical if < 1 slot, else actual elapsed wall-clock
                inst = calculate_upload_latency(task.data_size_bits, max(rate, 1e-12))
                task.t_upload = inst if inst <= SLOT_DURATION else task.upload_elapsed
                task.enqueue_time = now
                task.status = "queued"
                uav.active_upload = None
                if task in user.pending_tasks:
                    user.pending_tasks.remove(task)

                # Phase-3: Enqueue for Stage 0 on its destination UAV (e.g. UAV0 for 'A')
                st0 = task.stages[0]
                st0.enqueue_time = now
                st0.status = "queued"
                self.uavs[st0.uav_id].task_queue.append(task)

    def _process_compute(self, now, e_comp):
        stages_to_forward = []

        for uav in self._active_uavs():
            if uav.battery_energy <= 0.0:
                continue

            cycles_left = uav.cpu_freq * SLOT_DURATION
            queue_len_start = len(uav.task_queue)
            tasks_popped = 0

            while cycles_left > 1e-6:
                if uav.computing is None:
                    if tasks_popped >= queue_len_start or not uav.task_queue:
                        break
                    task = uav.task_queue.popleft()
                    tasks_popped += 1
                    st = task.current_stage
                    if st is None:
                        continue
                    st.compute_start_time = now
                    task.allocated_cpu_hz = uav.cpu_freq
                    if task.current_stage_idx == 0:
                        task.compute_start_time = now
                        upload_buffer_wait = max(
                            0.0,
                            (task.upload_start_time - task.arrival_time)
                            if task.upload_start_time is not None else 0.0,
                        )
                        cpu_queue_wait = max(
                            0.0,
                            (st.compute_start_time - st.enqueue_time)
                            if st.enqueue_time is not None else 0.0,
                        )
                        st.upload_buffer_wait = upload_buffer_wait
                        st.queue_wait = cpu_queue_wait
                        st.t_queue = upload_buffer_wait + cpu_queue_wait
                    else:
                        st.upload_buffer_wait = 0.0
                        st.queue_wait = max(
                            0.0,
                            (st.compute_start_time - st.enqueue_time)
                            if st.enqueue_time is not None else 0.0,
                        )
                        st.t_queue = st.queue_wait
                    st.status = "computing"
                    task.status = "computing"
                    uav.computing = task

                task = uav.computing
                st = task.current_stage
                if st is None:
                    uav.computing = None
                    continue

                processed = min(st.remaining_cycles, cycles_left)
                st.remaining_cycles = max(0.0, st.remaining_cycles - processed)
                task.remaining_cycles = max(0.0, task.remaining_cycles - processed)
                st.compute_elapsed += SLOT_DURATION
                task.compute_elapsed += SLOT_DURATION
                cycles_left -= processed
                energy = computation_energy(uav.cpu_freq, processed)
                e_comp[uav.id] += energy
                st.energy += energy

                if st.remaining_cycles <= 1e-6:
                    st.remaining_cycles = 0.0
                    st.compute_finish_time = now
                    closed = calculate_compute_latency(st.cpu_cycles, uav.cpu_freq)
                    st.t_compute = closed if closed <= SLOT_DURATION else st.compute_elapsed
                    st.status = "completed"
                    uav.computing = None

                    # Advance stage
                    task.current_stage_idx += 1
                    if task.current_stage_idx < len(task.stages):
                        next_st = task.stages[task.current_stage_idx]
                        next_st.enqueue_time = now
                        next_st.status = "queued"
                        task.status = "queued"
                        stages_to_forward.append((next_st.uav_id, task))
                    else:
                        # Chain completed
                        task.compute_finish_time = now
                        task.t_queue = sum(s.t_queue for s in task.stages)
                        task.t_compute = sum(s.t_compute for s in task.stages)
                        task.t_a2a = calculate_a2a_latency()
                        task.t_total = calculate_total_latency(
                            task.t_upload, task.t_queue, task.t_compute, task.t_a2a)
                        task.deadline_missed = (task.t_total > task.deadline)
                        if task.deadline_missed:
                            self._slot_deadline_misses += 1
                        task.status = "completed"
                        self.completed_tasks.append(task)
                        self._slot_completed.append(task)
                else:
                    break

        # 3. Transfer completed stages to destination queues (available next slot)
        for dest_uav_id, task in stages_to_forward:
            self.uavs[dest_uav_id].task_queue.append(task)

    def _pack_reward_info(self, e_flight, e_comm, e_comp, e_total_vec):
        completed = self._slot_completed
        if completed:
            mean_upload = float(np.mean([t.t_upload for t in completed]))
            mean_queue  = float(np.mean([t.t_queue  for t in completed]))
            mean_comp   = float(np.mean([t.t_compute for t in completed]))
            mean_total  = float(np.mean([t.t_total  for t in completed]))
            mean_rate   = float(np.mean([t.last_a2g_rate     for t in completed]))
            mean_dist   = float(np.mean([t.last_a2g_distance for t in completed]))
        else:
            mean_upload = mean_queue = mean_comp = mean_total = 0.0
            mean_rate = mean_dist = 0.0

        fleet_energy = float(np.sum(e_total_vec[:self.num_active]))
        # Reward: -(α L̃ + β Ẽ)
        lat_term = mean_total / PHASE1_LAT_NORM
        en_term  = fleet_energy / PHASE1_ENERGY_NORM
        global_reward = -(ALPHA_LATENCY * lat_term + BETA_ENERGY * en_term)
        rewards = np.full(MAX_UAVS, global_reward, dtype=np.float64)
        rewards[self.num_active:] = 0.0

        # Phase-2: percentile latency from episode history
        ep_lats = self._episode_latencies
        if ep_lats:
            arr = np.array(ep_lats, dtype=float)
            p50 = float(np.percentile(arr, 50))
            p95 = float(np.percentile(arr, 95))
            p99 = float(np.percentile(arr, 99))
            chain_avg_lat = float(np.mean(arr))
        else:
            p50 = p95 = p99 = 0.0
            chain_avg_lat = 0.0

        # Phase-2: pending tasks across all active UAVs
        tasks_pending = int(sum(
            len(u.upload_buffer) + int(u.active_upload is not None) +
            len(u.task_queue) + int(u.computing is not None)
            for u in self._active_uavs()
        ))

        batteries  = [float(u.battery_energy) for u in self._active_uavs()]
        queue_lens = []
        for u in self._active_uavs():
            backlog = (len(u.upload_buffer) + len(u.task_queue)
                       + int(u.active_upload is not None)
                       + int(u.computing is not None))
            queue_lens.append(backlog)

        avg_queue_len = float(np.mean(queue_lens)) if queue_lens else 0.0
        max_queue_len = int(max(queue_lens)) if queue_lens else 0

        e_fl  = float(np.sum(e_flight[:self.num_active]))
        e_co  = float(np.sum(e_comm[:self.num_active]))
        e_cp  = float(np.sum(e_comp[:self.num_active]))

        info = {
            # ---- slot-level task counters (Phase-1 + Phase-2 keys) ----
            "num_tasks_generated": self._slot_generated,
            "num_tasks_completed": len(completed),
            "tasks_generated":     self._slot_generated,
            "tasks_completed":     len(completed),
            "tasks_pending":       tasks_pending,
            "deadline_misses":     self._slot_deadline_misses,
            # ---- episode-level counters (Phase-2) ----
            "episode_tasks_generated":  self._episode_tasks_generated,
            "episode_tasks_completed":  self._episode_tasks_completed,
            "episode_deadline_misses":  self._episode_deadline_misses,
            # ---- Phase-3: chain metrics ----
            "chains_completed":         len(completed),
            "episode_chains_completed": self._episode_tasks_completed,
            "chain_avg_latency":        chain_avg_lat,
            "chain_p50_latency":        p50,
            "chain_p95_latency":        p95,
            "chain_p99_latency":        p99,
            # ---- latency breakdown (slot average) ----
            "upload_latency":  mean_upload,
            "queue_latency":   mean_queue,
            "compute_latency": mean_comp,
            "a2a_latency":     0.0,
            "total_latency":   mean_total,
            "avg_latency":     mean_total,
            # ---- percentile latency (episode, Phase-2) ----
            "p50_latency": p50,
            "p95_latency": p95,
            "p99_latency": p99,
            # ---- queue stats ----
            "queue_length":     queue_lens,
            "avg_queue_length": avg_queue_len,
            "max_queue_length": max_queue_len,
            # ---- energy (both naming conventions kept for compatibility) ----
            "energy_consumed":       fleet_energy,
            "energy_flight":         e_fl,
            "energy_communication":  e_co,
            "energy_computation":    e_cp,
            "flight_energy":         e_fl,
            "communication_energy":  e_co,
            "computation_energy":    e_cp,
            "total_energy":          fleet_energy,
            "total_energy_uav":      fleet_energy,
            # ---- battery ----
            "battery": batteries,
            # ---- A2G channel ----
            "a2g_rate":     mean_rate,
            "a2g_distance": mean_dist,
            # ---- simulation state ----
            "time_slot":       self.t,
            "num_active_uavs": self.num_active,
            # ---- compatibility stubs (DMJO legacy) ----
            "noma_pairs_formed": 0,
            "jammer_blocked":    0,
            "wind_speed":        0.0,
            "wind_direction":    0.0,
            "cache_local_hits":  0,
            "cache_coop_hits":   0,
            "cache_bs_fetches":  0,
        }
        return rewards, info
