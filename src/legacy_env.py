"""
LEGACY DMJO environment (archived, not used by Phase 1).

Original Diffusion-MAPPO multi-UAV MEC with NOMA, jammers, wind,
Gauss-Markov IoT mobility, cooperative caching, and two-timescale control.

Kept for later phases / comparison.  Do not import this from Phase-1 code.
The active environment is src/env.py.

================================================================
Original docstring
================================================================
Multi-Agent Environment for Diffusion-MAPPO
=============================================
CTDE (Centralised Training, Decentralised Execution) environment.

Each UAV agent receives:
  - Local observation  : own position, nearby devices, local latency, jammer info
  - Global state       : full system state (used only by centralised critic)

NOMA (Non-Orthogonal Multiple Access) is used so that each SUAV serves TWO IoT
devices per time slot simultaneously (full slot for uplink, no WPT phase):
  - Far user  (weaker channel) receives higher power fraction  α
  - Near user (stronger channel) receives fraction (1-α); decodes via SIC
  - Power split α is computed optimally each slot via bisection [P2] Eqs.(37)-(43)

Jammer interference is folded into the effective noise PSD passed to both
NOMA rate expressions, so secure rates degrade under jamming naturally.

Domain randomisation applied at configurable intervals to improve robustness.

References:
  [P2] Yin et al.: Multi-layer UAV architecture, NOMA Eqs.(20)-(21),(29),(37)-(43)
  [P4] Barman et al.: Task model, offloading decisions
  [P6] Liang et al.: UAV energy, Latency, time-slotted operation
"""

import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import *
# Ablation flags — imported explicitly so they survive the wildcard above
# and so the names are clearly visible in this file's namespace.
from config import (
    USE_LEARNED_TRAJ,   # False → fixed circular UAV orbit  (row 4)
    USE_COOP_CACHE,     # False → suppress coop reward signal (rows 5, 6)
    CACHE_TIMESCALE_K_C, # =1  → single-timescale cache updates (row 2)
)
# Gauss-Markov IoT mobility constants (Paper Section III.B, Eqs. 1–4)
try:
    _GM_RHO_V          = GM_RHO_V
    _GM_RHO_THETA      = GM_RHO_THETA
    _GM_MEAN_SPEED     = GM_MEAN_SPEED
    _GM_SIGMA_V        = GM_SIGMA_V
    _GM_SIGMA_THETA    = GM_SIGMA_THETA
    _IOT_WIND_DRIFT_FACTOR = IOT_WIND_DRIFT_FACTOR
except NameError:
    # Graceful fallback if running with an older config
    _GM_RHO_V          = 0.9
    _GM_RHO_THETA      = 0.9
    _GM_MEAN_SPEED     = 2.0
    _GM_SIGMA_V        = 0.5
    _GM_SIGMA_THETA    = 0.524
    _IOT_WIND_DRIFT_FACTOR = 0.12
from src.channel_model import ChannelModel, NOMACommunication
from src.caching import CooperativeCacheManager


# ================================================================
# ENTITY CLASSES
# ================================================================
class IoTDevice:
    def __init__(self, device_id, position, cpu_freq=IOT_CPU_FREQUENCY,
                 tx_power=IOT_TX_POWER, energy_max=IOT_ENERGY_BUFFER_MAX):
        self.id = device_id
        self.position = np.array(position, dtype=float)
        self.cpu_freq = cpu_freq
        self.tx_power = tx_power
        self.tx_power_base = tx_power
        self.energy_max = energy_max
        self.energy = energy_max
        self.latency = 0.0              # end-to-end task latency (seconds)
        self.served_this_slot = False    # flag for caching attribution
        self.cluster_id = -1

        # Gauss-Markov mobility state (Paper Section III.B, Eqs. 1–4)
        # Speed (m/slot) and heading (rad) evolve as AR-1 processes.
        self.prev_position = np.array(position, dtype=float)
        self.gm_speed      = 2.0          # current speed v_n(k)
        self.gm_heading    = 0.0          # current heading θ_n(k)
        self.gm_mean_heading = 0.0        # per-device mean heading θ̄_n
        # vx/vy kept for backward-compat with any code that reads velocity
        self.vx = 0.0
        self.vy = 0.0

        # Task model
        self.current_task = None

    def reset(self):
        self.energy = self.energy_max
        self.latency = 0.0
        self.served_this_slot = False
        self.current_task = None


class UAV:
    def __init__(self, uav_id, position, profile=None,
                 cpu_freq=2e9, tx_power=0.01, bandwidth=10e6,
                 cache_capacity=100, max_displacement=50,
                 max_concurrent=15):
        self.id = uav_id
        self.position = np.array(position, dtype=float)
        self.cpu_freq = cpu_freq
        self.tx_power = tx_power
        self.bandwidth = bandwidth
        self.cache_capacity = cache_capacity
        self.max_displacement = max_displacement
        self.max_concurrent = max_concurrent
        self.cached_files = []
        self.task_queue = []
        self.total_energy = 0.0

        # Macro-timescale velocity (m/slot) — updated at macro-interval
        # boundaries, applied every slot for smooth continuous trajectories.
        self.macro_vx = 0.0
        self.macro_vy = 0.0
        self.macro_vz = 0.0

        # Aliases read by observation and plotting code
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        if profile is not None:
            self.cpu_freq = profile['cpu_freq']
            self.cache_capacity = profile['cache_capacity']
            self.bandwidth = profile['bandwidth']
            self.tx_power = dbm_to_watt(profile['tx_power_dbm'])
            self.max_displacement = profile['max_displacement']
            self.max_concurrent = profile['max_concurrent']
            self.position[2] = profile['altitude']

        self.max_speed_per_slot = (self.max_displacement / max(CACHE_TIMESCALE_K_C, 1)) * 2.0

    def reset(self, position=None):
        if position is not None:
            self.position = np.array(position, dtype=float)
        self.task_queue = []
        self.total_energy = 0.0
        self.macro_vx = 0.0
        self.macro_vy = 0.0
        self.macro_vz = 0.0
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.max_speed_per_slot = (self.max_displacement / max(CACHE_TIMESCALE_K_C, 1)) * 2.0

    def get_capability_vector(self):
        """Normalised capability vector for observation (5 dims)."""
        return np.array([
            self.cpu_freq / 3e9,
            self.cache_capacity / 150,
            self.bandwidth / 15e6,
            self.tx_power / dbm_to_watt(27),
            self.position[2] / 120.0,
        ], dtype=np.float32)


class Jammer:
    def __init__(self, position, tx_power=JAMMER_TX_POWER):
        self.position = np.array(position, dtype=float)
        self.tx_power = tx_power


# ================================================================
# DOMAIN RANDOMISATION SAMPLER
# ================================================================
class DomainRandomiser:
    """
    Samples environment configurations from randomisation ranges
    defined in config.py.  Called periodically during training.
    """

    def __init__(self, rng=None):
        self.rng = rng or np.random.RandomState()

    def sample(self):
        """Return a dict of randomised parameters."""
        cfg = {}
        cfg['num_users']        = self.rng.randint(*RAND_NUM_USERS)
        cfg['num_uavs']         = self.rng.randint(*RAND_NUM_UAVS)
        cfg['num_jammers']      = self.rng.randint(*RAND_NUM_JAMMERS)
        cfg['jammer_tx_power']  = dbm_to_watt(self.rng.uniform(*RAND_JAMMER_POWER_DBM))
        cfg['area_radius']      = self.rng.uniform(*RAND_AREA_RADIUS)
        cfg['iot_tx_power']     = dbm_to_watt(self.rng.uniform(*RAND_IOT_TX_POWER_DBM))
        cfg['noise_psd']        = dbm_to_watt(self.rng.uniform(*RAND_NOISE_PSD_DBM))
        cfg['wind_speed']       = self.rng.uniform(*RAND_WIND_SPEED)
        cfg['jammer_interference_radius'] = self.rng.uniform(*RAND_JAMMER_INTERFERENCE_RADIUS)
        return cfg

    def default(self):
        """Return default (non-randomised) configuration."""
        return {
            'num_users':       NUM_USERS,
            'num_uavs':        NUM_UAVS,
            'num_jammers':     NUM_JAMMERS,
            'jammer_tx_power': JAMMER_TX_POWER,
            'area_radius':     AREA_RADIUS,
            'iot_tx_power':    IOT_TX_POWER,
            'noise_psd':       NOISE_PSD,
            'wind_speed':      WIND_SPEED_MEAN,
            'jammer_interference_radius': JAMMER_INTERFERENCE_RADIUS,
        }


# ================================================================
# MULTI-AGENT ENVIRONMENT
# ================================================================
class MultiUAVMECEnv:
    """
    CTDE Multi-Agent environment for MAPPO.

    n_agents = NUM_UAVS  (each heterogeneous UAV is an agent).
    All UAVs are peer agents with different capabilities.

    Per time slot, each SUAV runs NOMA Uplink Collection for the full slot:

      NOMA Uplink Collection
        Duration : SLOT_DURATION
        Coverage : TWO devices transmit simultaneously using NOMA:
                     Far user  (α · P_uav downlink power → stronger UL signal weight)
                     Near user ((1-α) · P_uav) — SIC decodes far user first, removes
                               its interference, then decodes near user cleanly.
        Jammer   : aggregate interference folded into effective noise PSD.
        Fallback : if cluster has only 1 device, falls back to single-user Shannon.

    Action space per agent — 3 continuous values in [-1, 1]:
      a[0]  dx        normalised x-displacement (× per-UAV max_displacement)
      a[1]  dy        normalised y-displacement (× per-UAV max_displacement)
      a[2]  schedule  continuous index into cluster device list → primary device

    NOMA power allocation is computed analytically per slot via the N-user
    golden-section optimiser in NOMACommunication.serve_cluster(), which
    minimises total cluster transmission delay. This decouples communication
    from the RL policy and reduces action dimensionality (Paper §IV.B).
    """

    def __init__(self, num_users=NUM_USERS, num_uavs=NUM_UAVS,
                 num_jammers=NUM_JAMMERS, seed=42, domain_cfg=None):
        self.rng = np.random.RandomState(seed)
        self.base_seed = seed

        # Apply domain config or defaults
        self.domain_cfg    = domain_cfg or DomainRandomiser(self.rng).default()
        self.num_users     = self.domain_cfg.get('num_users', num_users)
        self.num_uavs      = self.domain_cfg.get('num_uavs', num_uavs)
        self.num_uavs      = int(np.clip(self.num_uavs, MIN_UAVS, MAX_UAVS))
        self.n_agents      = MAX_UAVS    # ALWAYS pad to MAX_UAVS for fixed NN dims
        self.num_active     = self.num_uavs  # actual active UAVs this episode
        self.num_jammers   = self.domain_cfg.get('num_jammers', num_jammers)
        self.area_radius   = self.domain_cfg.get('area_radius', AREA_RADIUS)
        self.wind_speed    = self.domain_cfg.get('wind_speed', WIND_SPEED_MEAN)
        self.jammer_interference_radius = self.domain_cfg.get(
            'jammer_interference_radius', JAMMER_INTERFERENCE_RADIUS)

        # Wind state
        self.wind_direction = self.rng.uniform(0, 2 * np.pi)

        self.channel = ChannelModel()
        self.noma    = NOMACommunication(self.channel)
        # Cache manager with per-UAV capacities from profiles
        capacities = [UAV_PROFILES[i % len(UAV_PROFILES)]['cache_programs']
                      for i in range(self.num_uavs)]
        self.cache_manager = CooperativeCacheManager(
            num_uavs=self.num_uavs, per_uav_capacities=capacities,
            seed=seed + 7777)
        self.t = 0

        # ── Two-timescale hierarchical MDP tracking (Paper Section IV.B) ────
        # The macro-interval length K_c (= CACHE_TIMESCALE_K_C) separates:
        #   Macro-timescale: UAV trajectory + cooperative caching (every K_c slots)
        #   Micro-timescale: task offloading + NOMA power allocation (every slot)
        # UAV positions are FROZEN between macro boundaries; only offloading and
        # NOMA decisions change at each micro-epoch.
        self.macro_t          = 0          # current macro-interval index t
        self.slot_in_macro    = 0          # micro-epoch within current macro-interval
        # Per-slot propulsion energy (computed at macro step, spread over K_c slots)
        # so that the per-step reward signal reflects the ongoing cost of the macro
        # trajectory decision throughout the entire macro-interval.
        self._uav_prop_per_slot = np.zeros(MAX_UAVS, dtype=np.float64)

        self._init_entities()

        # Dimensions are FIXED at MAX_UAVS / MAX_NUM_USERS for stable NN
        self.max_cluster_size = (MAX_NUM_USERS // MIN_UAVS) + 5
        self.local_obs_dim    = self._compute_local_obs_dim()
        self.global_state_dim = self._compute_global_state_dim()
        self.agent_action_dim = self._compute_agent_action_dim()

        # Legacy compatibility
        self.state_dim  = self.global_state_dim
        self.action_dim = self.agent_action_dim * self.n_agents

    # ----------------------------------------------------------
    # Entity initialisation
    # ----------------------------------------------------------
    def _init_entities(self):
        cfg = self.domain_cfg

        # IoT devices — VARIABLE count, sampled each reset
        self.devices = []
        for i in range(self.num_users):
            angle = self.rng.uniform(0, 2 * np.pi)
            r = self.area_radius * np.sqrt(self.rng.uniform())
            pos = (r * np.cos(angle), r * np.sin(angle), 0)
            dev = IoTDevice(i, pos)
            dev.tx_power = cfg.get('iot_tx_power', IOT_TX_POWER)
            # Initialise Gauss-Markov mobility state
            dev.gm_speed        = _GM_MEAN_SPEED + self.rng.randn() * _GM_SIGMA_V
            dev.gm_speed        = max(dev.gm_speed, 0.1)
            dev.gm_heading      = self.rng.uniform(0, 2 * np.pi)
            dev.gm_mean_heading = dev.gm_heading  # per-device mean heading
            dev.prev_position   = dev.position.copy()
            dev.vx = dev.gm_speed * np.cos(dev.gm_heading)
            dev.vy = dev.gm_speed * np.sin(dev.gm_heading)
            self.devices.append(dev)

        # UAVs — heterogeneous profiles, equally spaced on inner circle
        self.uavs = []
        for i in range(self.num_uavs):
            profile = UAV_PROFILES[i % len(UAV_PROFILES)]
            angle = 2 * np.pi * i / self.num_uavs
            r = self.area_radius * 0.5
            alt = profile['altitude']
            pos = (r * np.cos(angle), r * np.sin(angle), alt)
            self.uavs.append(UAV(i, pos, profile=profile))

        # Backwards compat alias

        # Jammers — place centres using polar sampling so they always land
        # inside the circular deployment area.
        # The old code used uniform(-R, R) on both axes (square sampling):
        # up to 21% of placements fell OUTSIDE the circle, so the interference
        # zone barely overlapped any IoT devices → dev_jammed always False
        # → no X markers ever drawn in the plots.
        # We also cap the radial offset so the entire interference disc fits
        # inside the boundary: r_max = area_radius - jammer_interference_radius.
        jammer_power = cfg.get('jammer_tx_power', JAMMER_TX_POWER)
        self.jammers = []
        jam_r = self.jammer_interference_radius
        r_max = max(self.area_radius * 0.70, self.area_radius - jam_r)
        for _ in range(self.num_jammers):
            angle = self.rng.uniform(0, 2 * np.pi)
            r     = self.area_radius * 0.25 + self.rng.uniform(0, 1) * (r_max - self.area_radius * 0.25)
            jx    = r * np.cos(angle)
            jy    = r * np.sin(angle)
            self.jammers.append(Jammer((jx, jy, 0), tx_power=jammer_power))

        self._assign_clusters()

    def _update_device_positions(self):
        """
        Advance every IoT device by one slot using the Gauss-Markov mobility
        model (Paper Section III.B, Eqs. 1–4).

        Speed and heading each evolve as a first-order autoregressive process:

          v_n(k) = ρ_v · v_n(k-1)  +  (1-ρ_v) · v̄
                   +  √(1-ρ_v²) · σ_v · ξ^v(k)

          θ_n(k) = ρ_θ · θ_n(k-1)  +  (1-ρ_θ) · θ̄_n
                   +  √(1-ρ_θ²) · σ_θ · ξ^θ(k)

        where ξ ~ N(0,1) are independent Gaussian samples and
        ρ ∈ [0,1] is the memory parameter (0 = memoryless, 1 = constant).

        Position update (+ optional wind drift on lightweight nodes):
          x_n(k+1) = x_n(k) + v_n(k)·cos(θ_n(k)) + Δ_wind_x
          y_n(k+1) = y_n(k) + v_n(k)·sin(θ_n(k)) + Δ_wind_y

        Boundary handling: soft reflection — when a device reaches the
        deployment boundary the heading is mirrored and the device is
        clamped inside, so it curves back naturally without piling up.
        """
        R      = self.area_radius
        rho_v  = _GM_RHO_V
        rho_th = _GM_RHO_THETA
        v_bar  = _GM_MEAN_SPEED
        sig_v  = _GM_SIGMA_V
        sig_th = _GM_SIGMA_THETA
        noise_scale_v  = np.sqrt(1.0 - rho_v  ** 2) * sig_v
        noise_scale_th = np.sqrt(1.0 - rho_th ** 2) * sig_th

        for dev in self.devices:
            dev.prev_position[:] = dev.position

            # Save current (k-1) values before updating — position update
            # uses previous-step speed and heading per paper Eqs. 3-4:
            #   x_n(k) = x_n(k-1) + Δt · v_n(k-1) · cos(θ_n(k-1))
            prev_speed   = dev.gm_speed
            prev_heading = dev.gm_heading

            # ── Gauss-Markov speed update (Eq. 1) ────────────────────────────
            xi_v = self.rng.randn()
            new_speed = (rho_v * dev.gm_speed
                         + (1.0 - rho_v) * v_bar
                         + noise_scale_v * xi_v)
            dev.gm_speed = max(new_speed, 0.1)

            # ── Gauss-Markov heading update (Eq. 2) ──────────────────────────
            xi_th = self.rng.randn()
            new_heading = (rho_th * dev.gm_heading
                           + (1.0 - rho_th) * dev.gm_mean_heading
                           + noise_scale_th * xi_th)
            dev.gm_heading = new_heading % (2 * np.pi)

            # ── Position update (Eqs. 3–4) using k-1 values + wind drift ─────
            wind_drift_x = (_IOT_WIND_DRIFT_FACTOR
                            * self.wind_speed * np.cos(self.wind_direction))
            wind_drift_y = (_IOT_WIND_DRIFT_FACTOR
                            * self.wind_speed * np.sin(self.wind_direction))
            new_x = (dev.position[0]
                     + prev_speed * np.cos(prev_heading) + wind_drift_x)
            new_y = (dev.position[1]
                     + prev_speed * np.sin(prev_heading) + wind_drift_y)

            # ── Soft boundary reflection ──────────────────────────────────────
            r = np.sqrt(new_x ** 2 + new_y ** 2)
            if r > R * 0.90:
                # Mirror heading through the outward normal; bias mean heading
                # toward the area centre so the AR(1) process pulls the device
                # inward rather than orbiting along the boundary.
                outward_angle       = np.arctan2(new_y, new_x)
                dev.gm_heading      = (2.0 * outward_angle - dev.gm_heading) % (2 * np.pi)
                dev.gm_mean_heading = (outward_angle + np.pi) % (2 * np.pi)
                scale  = (R * 0.90) / r
                new_x *= scale
                new_y *= scale

            dev.position[0] = new_x
            dev.position[1] = new_y
            # Update vx/vy for any downstream code that reads velocity
            dev.vx = dev.gm_speed * np.cos(dev.gm_heading)
            dev.vy = dev.gm_speed * np.sin(dev.gm_heading)

    def _assign_clusters(self):
        """Assign each IoT device to the nearest active UAV (vectorised)."""
        if not self.devices or self.num_active == 0:
            return
        dev_pos = np.array([d.position[:2] for d in self.devices])       # (N, 2)
        uav_pos = np.array([u.position[:2] for u in self.uavs[:self.num_active]])  # (M, 2)
        # Broadcast distance: (N, 1, 2) - (1, M, 2) → (N, M)
        dists = np.linalg.norm(dev_pos[:, None, :] - uav_pos[None, :, :], axis=2)
        cluster_ids = np.argmin(dists, axis=1)
        for i, dev in enumerate(self.devices):
            dev.cluster_id = int(cluster_ids[i])

    # ----------------------------------------------------------
    # Observation / state dimensions
    # ----------------------------------------------------------
    def _compute_local_obs_dim(self):
        """
        Per-agent local observation (padded to MAX_UAVS):
          own position          (3: x, y, normalised altitude)
          own capabilities      (NUM_UAV_CAPABILITY_DIMS = 5)
          own velocity          (2)
          active_mask           (1)  — 1 if this agent is active, 0 if padded
          other UAVs relative   (2 * (MAX_UAVS - 1))
          other UAVs active     (MAX_UAVS - 1)  — mask for which are real
          cluster latencies          (K — padded)
          cluster energies      (K — padded)
          cluster jammer-blocked(K — padded)
          jammer relatives      (6 — padded to 3 jammers)
          jammer proximity      (3 — distance to each jammer, normalised)
          wind speed + dir      (2)
          cache state           (NUM_SERVICE_PROGRAMS)
          time fraction         (1)
        """
        K = self.max_cluster_size
        return (3 + NUM_UAV_CAPABILITY_DIMS + 2 + 1
                + 2 * (MAX_UAVS - 1) + (MAX_UAVS - 1)
                + K + K + K + 6 + 3 + 2 + NUM_SERVICE_PROGRAMS + 1)

    def _compute_global_state_dim(self):
        """
        Full centralised critic state (padded to MAX_UAVS & MAX_NUM_USERS):
          num_active_uavs       (1)
          all UAV positions     (2 * MAX_UAVS — padded)
          all UAV capabilities  (MAX_UAVS * NUM_UAV_CAPABILITY_DIMS)
          all UAV active masks  (MAX_UAVS)
          all device latencies       (MAX_NUM_USERS)
          all device energies   (MAX_NUM_USERS)
          all device jammed     (MAX_NUM_USERS)
          jammer positions      (6)
          wind speed + dir      (2)
          all cache states      (MAX_UAVS * NUM_SERVICE_PROGRAMS)
          time fraction         (1)
        """
        return (1 + 2 * MAX_UAVS
                + MAX_UAVS * NUM_UAV_CAPABILITY_DIMS
                + MAX_UAVS
                + MAX_NUM_USERS + MAX_NUM_USERS + MAX_NUM_USERS
                + 6 + 2
                + MAX_UAVS * NUM_SERVICE_PROGRAMS + 1)

    def _compute_agent_action_dim(self):
        """
        5-dimensional continuous action per agent (all in [-1, 1]):
          a[0]  dx       — normalised x-displacement (× max_speed_per_slot)
          a[1]  dy       — normalised y-displacement (× max_speed_per_slot)
          a[2]  dh       — altitude change, macro-only (× UAV_ALT_DELTA_MAX)
          a[3]  unused   — padding dimension
          a[4]  schedule — primary device index hint

        Altitude (a[2]) is applied only at macro-interval boundaries and
        clamped to [UAV_ALT_MIN, UAV_ALT_MAX]. Horizontal velocity (a[0], a[1])
        is updated at each macro step and applied every slot.
        noma_alpha is computed analytically via golden-section optimisation.
        """
        return 5

    # ----------------------------------------------------------
    # Jammer interference zone
    # ----------------------------------------------------------
    def is_in_jammer_zone(self, device):
        """Check if a single device is inside ANY jammer's interference zone."""
        for jammer in self.jammers:
            dist = np.linalg.norm(device.position[:2] - jammer.position[:2])
            if dist <= self.jammer_interference_radius:
                return True
        return False

    def _compute_jammed_flags_batch(self):
        """Vectorised jammer zone check for ALL devices at once."""
        if not self.jammers:
            return {d.id: False for d in self.devices}
        dev_pos = np.array([d.position[:2] for d in self.devices])  # (N, 2)
        jam_pos = np.array([j.position[:2] for j in self.jammers])  # (J, 2)
        # (N, J) distance matrix
        dists = np.linalg.norm(dev_pos[:, None, :] - jam_pos[None, :, :], axis=2)
        jammed = np.any(dists <= self.jammer_interference_radius, axis=1)  # (N,)
        return {self.devices[i].id: bool(jammed[i]) for i in range(len(self.devices))}

    def get_jammer_blocked_flags(self):
        """Return a list of 0/1 flags for each device (1 = blocked)."""
        return [1.0 if self.is_in_jammer_zone(d) else 0.0
                for d in self.devices]

    # ----------------------------------------------------------
    # Wind evolution
    # ----------------------------------------------------------
    def _evolve_wind(self):
        """
        Evolve wind speed AND direction each slot using
        Ornstein-Uhlenbeck process for realistic, smooth gusts.

        Speed: dS = θ(μ - S)dt + σ dW  → mean-reverting with random gusts
        Direction: random walk with small drift per slot
        """
        # Direction drift (random walk)
        self.wind_direction += (self.rng.randn()
                                * WIND_DIRECTION_CHANGE_RATE)
        self.wind_direction %= (2 * np.pi)

        # Speed — Ornstein-Uhlenbeck mean-reverting process
        mean = WIND_SPEED_MEAN
        theta = WIND_SPEED_REVERSION
        sigma = WIND_SPEED_VOLATILITY
        self.wind_speed += theta * (mean - self.wind_speed) + sigma * self.rng.randn()
        self.wind_speed = np.clip(self.wind_speed, WIND_SPEED_MIN, WIND_SPEED_MAX)

    # ----------------------------------------------------------
    # Task generation (fresh each slot, different every time)
    # ----------------------------------------------------------
    def _generate_tasks(self):
        """Generate random tasks for devices (vectorised batch sampling)."""
        N = len(self.devices)
        # Batch sample all random values at once
        probs      = self.rng.rand(N)
        types      = self.rng.randint(0, NUM_TASK_TYPES, N)
        complexity = (types + 1) / NUM_TASK_TYPES
        cycles     = self.rng.uniform(TASK_CPU_CYCLES_RANGE[0],
                                       TASK_CPU_CYCLES_RANGE[1], N) * complexity
        sizes      = self.rng.uniform(TASK_DATA_SIZE_RANGE[0],
                                       TASK_DATA_SIZE_RANGE[1], N) * complexity
        deadlines  = self.rng.uniform(*TASK_DEADLINE_RANGE, N)
        tx_powers  = self.rng.uniform(IOT_TX_POWER_MIN, IOT_TX_POWER_MAX, N)

        for i, dev in enumerate(self.devices):
            if probs[i] < TASK_GENERATION_PROB:
                dev.current_task = {
                    'task_type':  int(types[i]),
                    'cpu_cycles': cycles[i],
                    'data_size':  sizes[i],
                    'deadline':   deadlines[i],
                    'tx_power':   tx_powers[i],
                }
                dev.tx_power = tx_powers[i]
            else:
                dev.current_task = None

    def _try_local_compute(self, dev):
        """
        Check if the device can compute its task locally within the deadline.
        Returns (success: bool, delay: float, energy: float).
        Light tasks on a 0.5 GHz IoT CPU may finish in time.
        """
        task = dev.current_task
        if task is None:
            return False, 0.0, 0.0
        local_delay = task['cpu_cycles'] / dev.cpu_freq
        local_energy = EFFECTIVE_CAPACITANCE * (dev.cpu_freq ** 2) * task['cpu_cycles']
        threshold = task['deadline'] * LOCAL_COMPUTE_THRESHOLD_FACTOR
        can_compute = (local_delay <= threshold and local_energy <= dev.energy)
        return can_compute, local_delay, local_energy

    def _uav_compute(self, uav, task):
        """
        Compute task on UAV. Returns (delay, energy).
        Includes computation delay + result downlink delay.
        """
        comp_delay = task['cpu_cycles'] / uav.cpu_freq
        comp_energy = EFFECTIVE_CAPACITANCE * (uav.cpu_freq ** 2) * task['cpu_cycles']
        # Result delivery (downlink) — assume result is 10% of input data
        result_size = task['data_size'] * 0.1
        downlink_delay = result_size / max(uav.bandwidth * 0.5, 1e-6)
        total_delay = comp_delay + downlink_delay
        return total_delay, comp_energy

    # ----------------------------------------------------------
    # Observations (padded to MAX_UAVS)
    # ----------------------------------------------------------
    def get_local_obs(self, agent_id):
        """Build local observation vector for agent_id (padded to MAX_UAVS)."""
        R   = self.area_radius
        obs = []

        # Is this agent active? (inactive agents get all-zero obs)
        is_active = 1.0 if agent_id < self.num_active else 0.0

        if agent_id < self.num_active:
            uav = self.uavs[agent_id]
            # Own normalised position (x, y, altitude)
            alt_range = max(UAV_ALT_MAX - UAV_ALT_MIN, 1.0)
            obs.extend([uav.position[0] / R,
                        uav.position[1] / R,
                        (uav.position[2] - UAV_ALT_MIN) / alt_range])
            # Own capability vector
            obs.extend(uav.get_capability_vector().tolist())
            # Own velocity (fixes Markov property — SUG-5)
            obs.extend([uav.vx / uav.max_displacement,
                        uav.vy / uav.max_displacement])
        else:
            obs.extend([0.0] * (3 + NUM_UAV_CAPABILITY_DIMS + 2))

        # Active mask
        obs.append(is_active)

        # Relative positions of ALL other UAVs (padded to MAX_UAVS-1)
        for j in range(MAX_UAVS):
            if j == agent_id:
                continue
            if j < self.num_active and agent_id < self.num_active:
                other = self.uavs[j]
                uav = self.uavs[agent_id]
                obs.extend([(other.position[0] - uav.position[0]) / R,
                            (other.position[1] - uav.position[1]) / R])
            else:
                obs.extend([0.0, 0.0])

        # Active masks for other UAVs
        for j in range(MAX_UAVS):
            if j == agent_id:
                continue
            obs.append(1.0 if j < self.num_active else 0.0)

        # Cluster device latencies, energies, and jammer-blocked flags
        if agent_id < self.num_active:
            cluster_devs = [d for d in self.devices if d.cluster_id == agent_id]
        else:
            cluster_devs = []
        lats     = [d.latency / (SLOT_DURATION * NUM_TIME_SLOTS) for d in cluster_devs]
        energies = [d.energy / d.energy_max for d in cluster_devs]
        jammed   = [1.0 if self.is_in_jammer_zone(d) else 0.0
                     for d in cluster_devs]

        K        = self.max_cluster_size
        lats     = (lats     + [0.0] * K)[:K]
        energies = (energies + [0.0] * K)[:K]
        jammed   = (jammed   + [0.0] * K)[:K]
        obs.extend(lats)
        obs.extend(energies)
        obs.extend(jammed)

        # Jammer relative positions (zero-padded to 3 jammers)
        for j_idx in range(3):
            if j_idx < len(self.jammers) and agent_id < self.num_active:
                jm = self.jammers[j_idx]
                uav = self.uavs[agent_id]
                obs.extend([(jm.position[0] - uav.position[0]) / R,
                            (jm.position[1] - uav.position[1]) / R])
            else:
                obs.extend([0.0, 0.0])

        # Jammer proximity (normalised distance, for trajectory avoidance)
        for j_idx in range(3):
            if j_idx < len(self.jammers) and agent_id < self.num_active:
                jm = self.jammers[j_idx]
                uav = self.uavs[agent_id]
                dist = np.linalg.norm(jm.position[:2] - uav.position[:2])
                obs.append(np.clip(dist / (JAMMER_PROXIMITY_PENALTY_RADIUS * 2), 0, 1))
            else:
                obs.append(1.0)  # max distance = safe

        # Wind info
        obs.append(self.wind_speed / WIND_SPEED_MAX)
        obs.append(self.wind_direction / (2 * np.pi))

        # Cache state (padded to MAX_UAVS worth via cache manager)
        if agent_id < self.num_active:
            cache_state = self.cache_manager.get_cache_state(agent_id)
        else:
            cache_state = np.zeros(NUM_SERVICE_PROGRAMS)
        obs.extend(cache_state.tolist())

        # Time fraction
        obs.append(self.t / NUM_TIME_SLOTS)

        return np.array(obs, dtype=np.float32)

    def get_all_local_obs(self):
        """Return stacked local observations. Shape: (MAX_UAVS, local_obs_dim)."""
        return np.stack([self.get_local_obs(i) for i in range(MAX_UAVS)])

    def get_global_state(self):
        """Full state vector for the centralised critic (padded to MAX_UAVS & MAX_NUM_USERS)."""
        R     = self.area_radius
        state = []

        # Number of active UAVs (normalised)
        state.append(self.num_active / MAX_UAVS)

        # All UAV positions (padded to MAX_UAVS)
        for i in range(MAX_UAVS):
            if i < self.num_active:
                state.extend([self.uavs[i].position[0] / R,
                              self.uavs[i].position[1] / R])
            else:
                state.extend([0.0, 0.0])

        # All UAV capabilities (padded)
        for i in range(MAX_UAVS):
            if i < self.num_active:
                state.extend(self.uavs[i].get_capability_vector().tolist())
            else:
                state.extend([0.0] * NUM_UAV_CAPABILITY_DIMS)

        # Active masks
        for i in range(MAX_UAVS):
            state.append(1.0 if i < self.num_active else 0.0)

        # Device latencies, energies, jammed — padded to MAX_NUM_USERS
        lats = [d.latency / (SLOT_DURATION * NUM_TIME_SLOTS) for d in self.devices]
        lats = (lats + [0.0] * MAX_NUM_USERS)[:MAX_NUM_USERS]
        state.extend(lats)

        energies = [d.energy / d.energy_max for d in self.devices]
        energies = (energies + [0.0] * MAX_NUM_USERS)[:MAX_NUM_USERS]
        state.extend(energies)

        jammed = [1.0 if self.is_in_jammer_zone(d) else 0.0
                   for d in self.devices]
        jammed = (jammed + [0.0] * MAX_NUM_USERS)[:MAX_NUM_USERS]
        state.extend(jammed)

        for j_idx in range(3):
            if j_idx < len(self.jammers):
                state.extend([self.jammers[j_idx].position[0] / R,
                              self.jammers[j_idx].position[1] / R])
            else:
                state.extend([0.0, 0.0])

        state.append(self.wind_speed / WIND_SPEED_MAX)
        state.append(self.wind_direction / (2 * np.pi))

        # Cache states (padded to MAX_UAVS)
        for i in range(MAX_UAVS):
            if i < self.num_active:
                cache_state = self.cache_manager.get_cache_state(i)
                state.extend(cache_state.tolist())
            else:
                state.extend([0.0] * NUM_SERVICE_PROGRAMS)

        state.append(self.t / NUM_TIME_SLOTS)
        return np.array(state, dtype=np.float32)

    def get_state(self):
        return self.get_global_state()

    # ----------------------------------------------------------
    # Reset
    # ----------------------------------------------------------
    def reset(self, domain_cfg=None):
        """Reset environment, optionally applying a new domain config."""
        self.t = 0
        if domain_cfg is not None:
            self.domain_cfg    = domain_cfg
            self.num_users     = domain_cfg.get('num_users',     NUM_USERS)
            self.num_uavs      = int(np.clip(
                domain_cfg.get('num_uavs', NUM_UAVS), MIN_UAVS, MAX_UAVS))
            self.num_active    = self.num_uavs
            self.num_jammers   = domain_cfg.get('num_jammers',   NUM_JAMMERS)
            self.area_radius   = domain_cfg.get('area_radius',   AREA_RADIUS)
            self.wind_speed    = domain_cfg.get('wind_speed', WIND_SPEED_MEAN)
            self.jammer_interference_radius = domain_cfg.get(
                'jammer_interference_radius', JAMMER_INTERFERENCE_RADIUS)
        else:
            # Re-randomize IoT count and wind, but keep num_uavs from
            # the current domain_cfg to avoid unexpected UAV count changes
            # within a training run. UAV count only changes via domain_cfg.
            self.num_users  = self.rng.randint(*RAND_NUM_USERS)
            self.wind_speed = np.clip(
                WIND_SPEED_MEAN + self.rng.randn() * 1.0,
                WIND_SPEED_MIN, WIND_SPEED_MAX)
            # num_uavs / num_active stay as they were

        self.wind_direction = self.rng.uniform(0, 2 * np.pi)
        self._init_entities()

        # Reset or rebuild cache manager only if UAV count changed
        if (not hasattr(self, 'cache_manager') or
                self.cache_manager.num_uavs != self.num_uavs):
            capacities = [UAV_PROFILES[i % len(UAV_PROFILES)]['cache_programs']
                          for i in range(self.num_uavs)]
            self.cache_manager = CooperativeCacheManager(
                num_uavs=self.num_uavs, per_uav_capacities=capacities,
                seed=self.base_seed + 7777)
        else:
            self.cache_manager.reset()

        # Dimensions stay FIXED (MAX_UAVS / MAX_NUM_USERS padding)
        self.max_cluster_size = (MAX_NUM_USERS // MIN_UAVS) + 5
        self.local_obs_dim    = self._compute_local_obs_dim()
        self.global_state_dim = self._compute_global_state_dim()

        # ── Reset two-timescale counters ─────────────────────────────────────
        # macro_t is a global counter across all episodes — do NOT reset it here.
        # slot_in_macro is episode-local (position within current macro-interval).
        if not hasattr(self, 'macro_t'):
            self.macro_t = 0
        self.slot_in_macro       = 0
        self._uav_prop_per_slot  = np.zeros(MAX_UAVS, dtype=np.float64)

        obs   = self.get_all_local_obs()
        state = self.get_global_state()
        return obs, state

    # ----------------------------------------------------------
    # Step (multi-agent, NOMA Uplink)
    # ----------------------------------------------------------
    def step(self, actions):
        """
        Execute one time slot.

        Args:
            actions : np.array shape (n_agents, 5)
                      Each row: [dx, dy, dh, unused, schedule] — all in [-1, 1]
                        a[0] dx       — x-displacement (× max_speed_per_slot)
                        a[1] dy       — y-displacement (× max_speed_per_slot)
                        a[2] dh       — altitude change, macro-only (× UAV_ALT_DELTA_MAX)
                        a[3] unused   — padding dimension
                        a[4] schedule — device index hint

        Returns:
            next_obs  : (n_agents, local_obs_dim)
            next_state: (global_state_dim,)
            rewards   : (n_agents,)
            done      : bool
            info      : dict with per-slot diagnostics
        """
        self.t += 1
        R                = self.area_radius
        total_energy_uav = 0.0
        agent_rewards    = np.zeros(MAX_UAVS)

        # Two-timescale hierarchical MDP (Paper §IV.A–IV.B).
        # Macro boundary (every K_c slots): UAV velocity updated, caches refreshed.
        # Micro epochs (between boundaries): offloading and NOMA only; UAV moves
        # continuously using the stored macro velocity from the last boundary.
        K_c            = CACHE_TIMESCALE_K_C
        is_macro_step  = (self.t - 1) % K_c == 0
        self.slot_in_macro = (self.t - 1) % K_c
        # macro_t is a global counter — increment at each macro boundary.
        # It does NOT reset between episodes so training logs show monotonic progress.
        if is_macro_step and self.t > 0:
            self.macro_t += 1

        # ── Evolve wind ──
        self._evolve_wind()
        # Propagate current wind speed to channel model so A2G gains
        # reflect wind-induced UAV vibration / antenna misalignment.
        self.channel.wind_speed = self.wind_speed

        # ── Move IoT devices ──
        self._update_device_positions()

        # ── Generate fresh tasks (different every slot) ──
        self._generate_tasks()

        # Reset served flags for this slot
        for dev in self.devices:
            dev.served_this_slot = False

        # Jammer flags computed once (vectorised)
        jammed_flags = self._compute_jammed_flags_batch()

        scheduled_devices = []
        served_by         = []
        noma_pairs_formed = 0
        successful_tx     = 0
        jammer_blocked_count = 0
        local_compute_count  = 0
        uav_compute_count    = 0
        compute_fail_count   = 0

        # Only iterate over ACTIVE UAVs (inactive padded agents do nothing)
        for agent_id in range(self.num_active):
            uav = self.uavs[agent_id]
            a   = actions[agent_id]

            # Two-timescale trajectory control (Paper §IV.A–IV.B):
            # - Macro boundary: MAPPO updates stored velocity (macro_vx/vy/vz).
            # - Every slot: UAV moves by (macro_vx, macro_vy), giving smooth
            #   trajectories. Net displacement over K_c slots equals v_m·ΔT_c,
            #   matching paper eq. q_m(k+K_c) = q_m(k) + v_m(t)·ΔT_c.
            R_limit   = R * 0.88
            max_speed = uav.max_speed_per_slot

            if is_macro_step and USE_LEARNED_TRAJ:
                # Momentum blend: new_v = 0.65*old_v + 0.35*target_v
                momentum_alpha = 0.65
                dist_now = np.linalg.norm(uav.position[:2])
                if dist_now > R_limit * 0.80:
                    edge_frac = min(
                        (dist_now - R_limit * 0.80) / (R_limit * 0.20), 1.0)
                    momentum_alpha = max(momentum_alpha * (1.0 - 0.6 * edge_frac), 0.40)

                target_vx = float(a[0]) * max_speed
                target_vy = float(a[1]) * max_speed

                uav.macro_vx = (momentum_alpha * uav.macro_vx
                                + (1.0 - momentum_alpha) * target_vx)
                uav.macro_vy = (momentum_alpha * uav.macro_vy
                                + (1.0 - momentum_alpha) * target_vy)

                # Altitude update — macro-only, clamped to [UAV_ALT_MIN, UAV_ALT_MAX]
                dh = float(a[2]) * UAV_ALT_DELTA_MAX
                new_h = float(np.clip(uav.position[2] + dh, UAV_ALT_MIN, UAV_ALT_MAX))
                uav.macro_vz    = new_h - uav.position[2]
                uav.position[2] = new_h
                uav.vz          = uav.macro_vz

            elif is_macro_step and not USE_LEARNED_TRAJ:
                # Ablation row 4: fixed circular orbit
                orbit_r   = AREA_RADIUS * 0.5
                omega     = (2.0 * np.pi / NUM_TIME_SLOTS) * (1.0 + agent_id * 0.15)
                angle     = omega * self.t + agent_id * (
                    2.0 * np.pi / max(self.num_active, 1))
                target_x  = orbit_r * np.cos(angle)
                target_y  = orbit_r * np.sin(angle)
                diff      = np.array([target_x, target_y]) - uav.position[:2]
                dist_diff = np.linalg.norm(diff) + 1e-8
                spd       = min(max_speed, dist_diff)
                uav.macro_vx = (diff[0] / dist_diff) * spd
                uav.macro_vy = (diff[1] / dist_diff) * spd

            # Every slot: apply stored macro velocity + wind drift
            wind_vx = self.wind_speed * np.cos(self.wind_direction) * SLOT_DURATION
            wind_vy = self.wind_speed * np.sin(self.wind_direction) * SLOT_DURATION

            dx = uav.macro_vx + wind_vx
            dy = uav.macro_vy + wind_vy

            # Airspeed relative to air mass (for propulsion energy)
            v_air_x = uav.macro_vx - wind_vx
            v_air_y = uav.macro_vy - wind_vy

            target_pos  = uav.position[:2] + np.array([dx, dy])
            dist_target = np.linalg.norm(target_pos)

            if dist_target > R_limit:
                # Clamp to boundary and zero the outward velocity component
                target_pos = target_pos * (R_limit / dist_target)
                outward    = target_pos / (np.linalg.norm(target_pos) + 1e-9)
                vdot       = uav.macro_vx * outward[0] + uav.macro_vy * outward[1]
                if vdot > 0:
                    uav.macro_vx -= vdot * outward[0]
                    uav.macro_vy -= vdot * outward[1]
                    spd = np.sqrt(uav.macro_vx**2 + uav.macro_vy**2)
                    if spd > max_speed:
                        uav.macro_vx *= max_speed / spd
                        uav.macro_vy *= max_speed / spd

            uav.position[:2] = target_pos
            uav.vx = uav.macro_vx
            uav.vy = uav.macro_vy

            # Propulsion energy this slot
            v_air = np.sqrt(v_air_x**2 + v_air_y**2) / max(SLOT_DURATION, 1e-9)
            v_air = max(v_air, 0.01)
            P_fly = (
                P1_BLADE * (1 + 3 * v_air**2 / ROTOR_TIP_SPEED**2) +
                P2_INDUCED * (np.sqrt(MEAN_ROTOR_VELOCITY**2 +
                                      v_air**4 / 4) - v_air**2 / 2) ** 0.5 +
                0.5 * DRAG_RATIO * AIR_DENSITY * ROTOR_SOLIDITY *
                DISC_AREA * v_air**3
            )
            E_wind_drag = WIND_DRAG_QUADRATIC_COEFF * v_air**2
            E_prop_slot = (P_fly + E_wind_drag) * SLOT_DURATION
            uav.total_energy += E_prop_slot
            total_energy_uav += E_prop_slot

            # ----------------------------------------
            # Full slot used for NOMA uplink (no WPT phase)
            # ----------------------------------------
            dt_duration = SLOT_DURATION   # Full slot for uplink

            # ----------------------------------------
            # Cluster devices for this agent
            # ----------------------------------------
            cluster_devs = [d for d in self.devices if d.cluster_id == agent_id]

            if not cluster_devs:
                scheduled_devices.append(None)
                # Propulsion energy already accounted for in the two-timescale
                # movement block above (either macro flight or micro hovering).
                continue

            # ----------------------------------------
            # NOMA Uplink
            # ----------------------------------------
            noise_psd = self.domain_cfg.get('noise_psd', NOISE_PSD)

            jammer_interference = sum(
                self.channel.a2g_channel_gain(j.position, uav.position) * j.tx_power
                for j in self.jammers
            )
            eff_noise_psd = noise_psd + jammer_interference / uav.bandwidth

            # Filter out jammer-blocked devices
            eligible_devs = []
            eligible_indices = []
            for idx, d in enumerate(cluster_devs):
                if jammed_flags[d.id]:
                    jammer_blocked_count += 1
                    d.latency += SLOT_DURATION         # blocked — latency still grows
                    agent_rewards[agent_id] -= 0.3  # mild penalty
                else:
                    eligible_devs.append(d)
                    eligible_indices.append(idx)

            # Pre-compute A2G channel gains (vectorised)
            if eligible_devs:
                elig_positions = np.array([d.position for d in eligible_devs])
                gains = self.channel.batch_a2g_gains(elig_positions, uav.position)
            else:
                gains = np.array([])

            # N-user SIC NOMA: serve ALL eligible devices simultaneously.
            # serve_cluster() runs golden-section power optimisation (Eq. 14),
            # computes per-user SIC SINR (Eq. 12), and returns per-user results.
            # a[4] (schedule hint) is retained in the action space but the
            # N-user formulation serves the entire cluster, not a single device.
            if eligible_devs:
                data_sizes = [d.current_task['data_size'] if d.current_task
                              else TASK_DATA_SIZE for d in eligible_devs]

                noma_results = self.noma.serve_cluster(
                    user_indices  = list(range(len(eligible_devs))),
                    channel_gains = gains.tolist(),
                    data_sizes    = data_sizes,
                    P_total       = uav.tx_power,
                    bandwidth     = uav.bandwidth,
                    noise_psd     = eff_noise_psd,
                )

                for res in noma_results:
                    dev        = eligible_devs[res['user_idx']]
                    bits_rx    = dt_duration * res['rate']
                    e_consumed = dt_duration * dev.tx_power

                    if bits_rx >= MIN_TRANSMISSION_BITS and e_consumed <= dev.energy:
                        dev.latency          = dt_duration
                        dev.served_this_slot = True
                        dev.energy          -= e_consumed
                        agent_rewards[agent_id] += 2.0
                        successful_tx += 1
                    else:
                        dev.latency += SLOT_DURATION
                        agent_rewards[agent_id] -= 0.5

                    scheduled_devices.append(dev)
                    served_by.append((dev, agent_id))

                if len(eligible_devs) >= 2:
                    noma_pairs_formed += 1
            # Propulsion energy already accumulated in the two-timescale
            # movement block (macro flight or micro hovering) above.

        # ----------------------------------------
        # Latency increment for all unserved devices
        # ----------------------------------------
        sched_set  = set(d.id for d in scheduled_devices if d is not None)
        jammed_set = {d_id for d_id, jammed in jammed_flags.items() if jammed}
        for dev in self.devices:
            if dev.id not in sched_set and dev.id not in jammed_set:
                dev.latency += SLOT_DURATION

        # Re-cluster
        self._assign_clusters()

        # ----------------------------------------
        # Task lifecycle: local compute vs UAV offload vs BS
        # ----------------------------------------
        self.cache_manager.clear_slot_links()
        cache_energy_cost = 0.0
        local_hits   = 0
        coop_hits    = 0
        bs_fetches   = 0
        bs_fetch_uavs = set()

        for dev, agent_id in served_by:
            if not dev.served_this_slot:
                continue

            task = dev.current_task
            if task is None:
                continue

            # Step 1: Can IoT compute locally?
            can_local, local_delay, local_energy = self._try_local_compute(dev)
            if can_local:
                dev.energy -= local_energy
                dev.energy = max(dev.energy, 0.0)
                dev.latency = local_delay          # E2E = just local compute
                agent_rewards[agent_id] += REWARD_LOCAL_COMPUTE
                local_compute_count += 1
                continue

            # Step 2: Offload to UAV — check service program cache
            required_prog = self.cache_manager.sample_required_program()
            tier, delay, partner_id = self.cache_manager.lookup(
                agent_id, required_prog)

            if tier == 1:
                uav = self.uavs[agent_id]
                comp_delay, comp_energy = self._uav_compute(uav, task)
                total_task_latency = dev.latency + comp_delay  # comm + compute + delivery
                if total_task_latency <= task['deadline']:
                    dev.latency = total_task_latency           # full E2E latency
                    agent_rewards[agent_id] += REWARD_UAV_COMPUTE + CACHE_HIT_REWARD
                    cache_energy_cost += comp_energy
                    uav_compute_count += 1
                else:
                    dev.latency = total_task_latency
                    agent_rewards[agent_id] += REWARD_COMPUTE_FAIL
                    compute_fail_count += 1
                local_hits += 1

            elif tier == 2:
                partner_uav = self.uavs[partner_id]
                comp_delay, comp_energy = self._uav_compute(partner_uav, task)
                a2a_energy = task['data_size'] * OFFLOAD_ENERGY_PER_BIT
                total_task_latency = dev.latency + delay + comp_delay  # comm + A2A + compute
                if total_task_latency <= task['deadline']:
                    dev.latency = total_task_latency
                    # Ablation rows 5 / 6: coop reward suppressed (USE_COOP_CACHE=False
                    # means tier==2 is never reached, but guard here for safety)
                    coop_bonus = CACHE_COOP_REWARD if USE_COOP_CACHE else 0.0
                    agent_rewards[agent_id] += REWARD_UAV_COMPUTE + coop_bonus
                    cache_energy_cost += comp_energy + a2a_energy
                    uav_compute_count += 1
                else:
                    dev.latency = total_task_latency
                    agent_rewards[agent_id] += REWARD_COMPUTE_FAIL
                    compute_fail_count += 1
                coop_hits += 1

            else:
                # Tier 3: BS fetch — service retrieval only (no BS-side computation)
                # BS acts as a cache server; latency = uplink propagation + backhaul fetch.
                # See Section III.C: "The BS does not participate in task computation."
                bs_total = dev.latency + OFFLOAD_DELAY_BS
                dev.latency = bs_total
                if bs_total <= task['deadline']:
                    agent_rewards[agent_id] += CACHE_MISS_PENALTY * 0.5
                else:
                    agent_rewards[agent_id] += CACHE_MISS_PENALTY
                    compute_fail_count += 1
                bs_fetches += 1
                bs_fetch_uavs.add(agent_id)

            self.cache_manager.log_task(
                agent_id, required_prog, completed=(tier <= 2))

        # Cache refresh aligned with macro boundary (Paper §IV.C).
        # Ablation row 2: CACHE_TIMESCALE_K_C=1 fires every slot.
        if is_macro_step:
            self.cache_manager.update_caches(self.t)
        total_energy_uav += cache_energy_cost

        # ----------------------------------------
        # Shaped rewards (zero-centred)
        # ----------------------------------------
        total_latency = sum(d.latency for d in self.devices)
        avg_latency   = total_latency / max(self.num_users, 1)
        # Baseline: theoretical steady-state latency
        baseline_latency = self.num_users / max(self.num_active * 2, 1)

        global_reward = -(ALPHA_LATENCY * (avg_latency - baseline_latency) +
                          BETA_ENERGY   * total_energy_uav / 1000)

        # Collision penalty (only among active UAVs)
        for i in range(self.num_active):
            for j in range(i + 1, self.num_active):
                dist = np.linalg.norm(
                    self.uavs[i].position[:2] - self.uavs[j].position[:2])
                if dist < 30:
                    agent_rewards[i] -= 5.0
                    agent_rewards[j] -= 5.0

            dist_from_centre = np.linalg.norm(self.uavs[i].position[:2])
            if dist_from_centre > R * 0.85:
                pen = 3.0 * ((dist_from_centre - R * 0.85) / (R * 0.15)) ** 2
                agent_rewards[i] -= pen

            # Jammer proximity penalty — UAV trajectory avoidance
            for jammer in self.jammers:
                jdist = np.linalg.norm(
                    self.uavs[i].position[:2] - jammer.position[:2])
                if jdist < JAMMER_PROXIMITY_PENALTY_RADIUS:
                    penalty = JAMMER_PROXIMITY_PENALTY_SCALE * (
                        1.0 - jdist / JAMMER_PROXIMITY_PENALTY_RADIUS)
                    agent_rewards[i] -= penalty

        rewards = 0.5 * agent_rewards + 0.5 * global_reward

        done = (self.t >= NUM_TIME_SLOTS)

        next_obs   = self.get_all_local_obs()
        next_state = self.get_global_state()

        # Build served_dev_ids (clear, no lambda)
        successful_pairs = [(d, a) for d, a in served_by if d.served_this_slot]
        served_dev_map = {}
        for dev, aid in successful_pairs:
            served_dev_map[dev.id] = {
                'agent': aid,
                'noma': sum(1 for _, a2 in successful_pairs if a2 == aid) >= 2,
            }

        info = {
            'total_latency':           total_latency,
            'avg_latency':             avg_latency,
            'total_energy_uav':    total_energy_uav,
            'successful_tx':       successful_tx,
            'noma_pairs_formed':   noma_pairs_formed,
            'jammer_blocked':      jammer_blocked_count,
            'time_slot':           self.t,
            'global_reward':       global_reward,
            'wind_speed':          self.wind_speed,
            'wind_direction':      self.wind_direction,
            'num_active_uavs':     self.num_active,
            # Task lifecycle stats
            'local_compute':       local_compute_count,
            'uav_compute':         uav_compute_count,
            'compute_fail':        compute_fail_count,
            # Caching stats
            'cache_local_hits':    local_hits,
            'cache_coop_hits':     coop_hits,
            'cache_bs_fetches':    bs_fetches,
            'bs_fetch_uavs':       list(bs_fetch_uavs),
            'cache_coverage':      self.cache_manager.get_collective_coverage(),
            'offload_links':       list(self.cache_manager.stats['offload_links']),
            'served_dev_ids':      served_dev_map,
            # Two-timescale diagnostics (Paper Section IV.A–IV.B)
            'is_macro_step':       is_macro_step,
            'macro_t':             self.macro_t,
            'slot_in_macro':       self.slot_in_macro,
            'macro_interval_len':  K_c,
        }

        return next_obs, next_state, rewards, done, info