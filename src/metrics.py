"""
Metrics & Cost Function Module
================================
Evaluation metrics and the multi-objective cost function.

Dead function removed:
  expected_user_delay() — referenced PUAV tiers (T_ps, T_rp, Pr_P) that
  no longer exist in this all-peer-UAV architecture.

References:
- Proposal Eq.(1): min Σ (α·L_total + β·E_UAV)
- [P4] Eq.(12): End-to-end latency
- [P6] Eqs.(11)-(12): Secure Latency
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ALPHA_LATENCY, BETA_ENERGY,
    P1_BLADE, P2_INDUCED, ROTOR_TIP_SPEED, MEAN_ROTOR_VELOCITY,
    DRAG_RATIO, AIR_DENSITY, ROTOR_SOLIDITY, DISC_AREA,
    EFFECTIVE_CAPACITANCE, MIN_TRANSMISSION_BITS,
)


# ================================================================
# COST FUNCTION  (Proposal Eq.1)
# ================================================================
def compute_cost(latency_total, energy_uav,
                 alpha=ALPHA_LATENCY, beta=BETA_ENERGY) -> float:
    """
    Weighted multi-objective cost.
    C(t) = α · L_total(t) + β · E_UAV(t)   [Proposal Eq.1]
    """
    return alpha * latency_total + beta * energy_uav


def compute_cumulative_cost(latency_history, energy_history,
                             alpha=ALPHA_LATENCY, beta=BETA_ENERGY) -> float:
    """
    Cumulative cost over time horizon T.
    Σ_{t=0}^{T} (α · L_total(t) + β · E_UAV(t))
    """
    return sum(compute_cost(l, e, alpha, beta)
               for l, e in zip(latency_history, energy_history))


# ================================================================
# END-TO-END LATENCY  [P4] Eq.(12)
# ================================================================
def end_to_end_latency(T_comm, T_exe, T_delivery) -> float:
    """
    Total task latency.
    L(t_i) = T_comm + T_exe + T_delivery  [P4] Eq.(12)
    """
    return T_comm + T_exe + T_delivery


def computation_delay(cpu_cycles, cpu_frequency) -> float:
    """T_exe = φ_cycles / f_CPU  [P5] Eq.(3)"""
    return cpu_cycles / max(cpu_frequency, 1e-6)


def transmission_delay(data_size_bits, rate_bps) -> float:
    """T_tx = δ / R  [P4] Eq.(5)"""
    return data_size_bits / max(rate_bps, 1e-6)


# ================================================================
# SECURE AoI  [P6] Eqs.(11)-(12)
# ================================================================
def update_latency(current_latency, bits_transmitted, energy_consumed,
               energy_available, min_bits=MIN_TRANSMISSION_BITS) -> int:
    """
    Update latency for a single device.
    Resets to 1 on successful transmission, else increments.
    [P6] Eq.(11)
    """
    if bits_transmitted >= min_bits and energy_consumed <= energy_available:
        return 1
    return current_latency + 1


def total_secure_latency(device_latencies) -> float:
    """Sum latency across all devices. [P6] Eq.(12)"""
    return sum(device_latencies)


# ================================================================
# UAV ENERGY CONSUMPTION  [P6] Eqs.(9)-(10)
# ================================================================
def uav_propulsion_energy(velocity, duration,
                           P1=P1_BLADE, P2=P2_INDUCED,
                           v_tip=ROTOR_TIP_SPEED,
                           v_hover=MEAN_ROTOR_VELOCITY) -> float:
    """
    Rotary-wing UAV propulsion energy. [P6] Eq.(9)
    """
    v = max(velocity, 0.01)
    P_fly = (
        P1 * (1 + 3 * v**2 / v_tip**2) +
        P2 * (np.sqrt(v_hover**2 + v**4 / 4) - v**2 / 2)**0.5 +
        0.5 * DRAG_RATIO * AIR_DENSITY * ROTOR_SOLIDITY * DISC_AREA * v**3
    )
    return P_fly * duration


def uav_transfer_energy(tx_power, duration) -> float:
    """Energy for RF energy transfer. [P6] Eq.(10)"""
    return tx_power * duration


def computation_energy(cpu_frequency, cpu_cycles,
                        capacitance=EFFECTIVE_CAPACITANCE) -> float:
    """DVFS computation energy. [P4] Eq.(16)"""
    return capacitance * (cpu_frequency ** 2) * cpu_cycles


# ================================================================
# EPISODE METRICS TRACKER
# ================================================================
class MetricsTracker:
    """Track and aggregate evaluation metrics across an episode."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.episode_rewards      = []
        self.episode_latencies         = []
        self.episode_energies     = []
        self.episode_cache_hits   = []
        self.episode_task_completions = []
        self.episode_costs        = []

    def log_step(self, reward, avg_latency, uav_energy,
                 cache_hit=0, task_done=0):
        self.episode_rewards.append(reward)
        self.episode_latencies.append(avg_latency)
        self.episode_energies.append(uav_energy)
        self.episode_cache_hits.append(cache_hit)
        self.episode_task_completions.append(task_done)
        self.episode_costs.append(compute_cost(avg_latency, uav_energy))

    def get_episode_summary(self) -> dict:
        def _mean(lst):
            return float(np.mean(lst)) if lst else 0.0

        return {
            'total_reward':          sum(self.episode_rewards),
            'avg_reward':            _mean(self.episode_rewards),
            'avg_latency':               _mean(self.episode_latencies),
            'total_energy':          sum(self.episode_energies),
            'avg_cache_hit':         _mean(self.episode_cache_hits),
            'task_completion_rate':  _mean(self.episode_task_completions),
            'total_cost':            sum(self.episode_costs),
        }
