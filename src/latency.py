"""
Modular latency helpers for the UAV-MEC simulator.

Phase 1:
    T_total = T_upload + T_queue + T_compute

Later phases will add:
    calculate_a2a_latency()
    calculate_chain_latency()
"""

import numpy as np


def calculate_a2g_distance(user_pos, uav_pos):
    """3D Euclidean distance between a ground user and a UAV."""
    u = np.asarray(user_pos, dtype=float)
    v = np.asarray(uav_pos, dtype=float)
    return float(np.linalg.norm(u - v))


def calculate_upload_latency(data_size_bits, rate_bps):
    """T_upload = data_size_bits / R_A2G."""
    return float(data_size_bits) / max(float(rate_bps), 1e-12)


def calculate_queue_latency(enqueue_time, compute_start_time):
    """T_queue = time the task waits on the UAV before CPU starts."""
    return max(0.0, float(compute_start_time) - float(enqueue_time))


def calculate_compute_latency(cpu_cycles, allocated_cpu_hz):
    """T_compute = cpu_cycles / f_allocated."""
    return float(cpu_cycles) / max(float(allocated_cpu_hz), 1e-12)


def calculate_a2a_latency(data_size_bits=0.0, rate_bps=1e9):
    """T_A2A = data_size_bits / R_A2A (Phase 4)."""
    if rate_bps is None or rate_bps <= 0:
        return 0.0
    return float(data_size_bits) / max(float(rate_bps), 1e-12)


def calculate_total_latency(t_upload, t_queue, t_compute, t_a2a=0.0):
    """End-to-end latency including A2A transmission."""
    return float(t_upload) + float(t_queue) + float(t_compute) + float(t_a2a)
