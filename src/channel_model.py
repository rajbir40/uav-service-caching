"""
Channel Model — Paper §III.C
==============================
Implements A2G, A2A, and Rician fading channel models with probabilistic LoS,
plus N-user SIC NOMA power allocation and scheduling.

References:
  Paper §III.C Eqs. (3)–(14): LoS probability, path loss, channel gain
  Paper §III.D Eqs. (12)–(14): NOMA SINR, SIC, power allocation
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    ENV_CONSTANT_A, ENV_CONSTANT_B, PATH_LOSS_EXPONENT,
    LOS_ATTENUATION, NLOS_ATTENUATION, RICIAN_K_FACTOR,
    WIND_K_DEGRADATION_FACTOR, CHANNEL_POWER_GAIN_REF,
    NOISE_PSD, RBS_BANDWIDTH,
    PAIRING_WEIGHT_CHANNEL, PAIRING_WEIGHT_FILE,
)


class ChannelModel:
    """Unified A2G/A2A channel model for the multi-UAV MEC system."""

    def __init__(self, a=ENV_CONSTANT_A, b=ENV_CONSTANT_B,
                 l=PATH_LOSS_EXPONENT, eta_los=LOS_ATTENUATION,
                 eta_nlos=NLOS_ATTENUATION, k_rician=RICIAN_K_FACTOR):
        self.a        = a
        self.b        = b
        self.l        = l
        self.eta_los  = eta_los
        self.eta_nlos = eta_nlos
        self.k_rician = k_rician
        # Updated each slot by env.step() before channel computations.
        # Higher wind speed reduces effective LoS probability (antenna jitter).
        self.wind_speed = 0.0

    def euclidean_distance(self, pos_a, pos_b):
        return np.linalg.norm(np.array(pos_a) - np.array(pos_b))

    def horizontal_distance(self, pos_a, pos_b):
        return np.sqrt((pos_a[0] - pos_b[0])**2 + (pos_a[1] - pos_b[1])**2)

    def elevation_angle_deg(self, pos_ground, pos_aerial):
        """Elevation angle θ = (180/π)·arctan(h / d_hor)  (Paper §III.C)"""
        d_hor = self.horizontal_distance(pos_ground, pos_aerial)
        h     = abs(pos_aerial[2] - pos_ground[2])
        if d_hor < 1e-6:
            return 90.0
        return (180.0 / np.pi) * np.arctan(h / d_hor)

    def los_probability(self, pos_ground, pos_aerial):
        """P_LoS = 1 / (1 + a·exp(−b·(θ − a)))  (Paper Eq. 5)"""
        theta = self.elevation_angle_deg(pos_ground, pos_aerial)
        return 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))

    def nlos_probability(self, pos_ground, pos_aerial):
        return 1.0 - self.los_probability(pos_ground, pos_aerial)

    def path_loss_los(self, distance):
        """L_LoS = η_LoS · d^(−l)"""
        return self.eta_los * (max(distance, 1e-6) ** (-self.l))

    def path_loss_nlos(self, distance):
        """L_NLoS = η_NLoS · d^(−l)"""
        return self.eta_nlos * (max(distance, 1e-6) ** (-self.l))

    def a2g_channel_gain(self, pos_ground, pos_aerial):
        """
        Effective A2G gain: g = P_LoS·L_LoS + P_NLoS·L_NLoS  (Paper Eq. 8).
        Wind speed degrades LoS probability via antenna misalignment factor.
        """
        d      = self.euclidean_distance(pos_ground, pos_aerial)
        q_los  = self.los_probability(pos_ground, pos_aerial)
        q_los *= 1.0 / (1.0 + WIND_K_DEGRADATION_FACTOR * self.wind_speed)
        return q_los * self.path_loss_los(d) + (1.0 - q_los) * self.path_loss_nlos(d)

    def batch_a2g_gains(self, dev_positions, uav_position):
        """
        Vectorised A2G gains from N devices to one UAV.

        Args:
            dev_positions : (N, 3) array
            uav_position  : (3,) array
        Returns:
            gains : (N,) array
        """
        dev   = np.asarray(dev_positions)
        uav   = np.asarray(uav_position)
        diffs = dev - uav[None, :]
        dists = np.maximum(np.linalg.norm(diffs, axis=1), 1e-6)
        d_hor = np.maximum(np.sqrt(diffs[:, 0]**2 + diffs[:, 1]**2), 1e-6)
        h     = np.abs(uav[2] - dev[:, 2])
        theta = (180.0 / np.pi) * np.arctan(h / d_hor)
        q_los = 1.0 / (1.0 + self.a * np.exp(-self.b * (theta - self.a)))
        q_los *= 1.0 / (1.0 + WIND_K_DEGRADATION_FACTOR * self.wind_speed)
        return q_los * self.eta_los * dists**(-self.l) + (1.0 - q_los) * self.eta_nlos * dists**(-self.l)

    def a2a_channel_gain(self, pos_uav1, pos_uav2):
        """A2A channel gain (pure LoS): g = η_LoS · d^(−l)  (Paper Eq. 10)"""
        return self.path_loss_los(self.euclidean_distance(pos_uav1, pos_uav2))

    def rician_channel_vector(self, pos_a, pos_b, k_factor=None, rng=None):
        """
        Rician fading channel gain |h|² (Paper §III.C).
        Accepts an optional seeded rng for reproducibility.
        """
        k_factor = k_factor if k_factor is not None else self.k_rician
        d        = max(self.euclidean_distance(pos_a, pos_b), 1e-6)
        large    = CHANNEL_POWER_GAIN_REF / (d ** self.l)
        _rng     = rng if rng is not None else np.random
        h_nlos   = (_rng.randn() + 1j * _rng.randn()) / np.sqrt(2)
        h = (np.sqrt(k_factor / (k_factor + 1)) +
             np.sqrt(1.0 / (k_factor + 1)) * h_nlos)
        return large * (np.abs(h) ** 2)

    def transmission_rate(self, tx_power, channel_gain, bandwidth,
                          noise_psd=NOISE_PSD, interference_power=0):
        """Shannon capacity: R = B·log₂(1 + P·g / (I + N₀·B))"""
        snr = tx_power * channel_gain / max(interference_power + noise_psd * bandwidth, 1e-20)
        return bandwidth * np.log2(1.0 + max(snr, 1e-10))

    def a2g_uplink_rate(self, pos_user, pos_uav, tx_power, bandwidth,
                        noise_psd=NOISE_PSD):
        """
        Single-user A2G uplink rate (Phase 1).

        SINR = P * g / (N0 * B),  R = B * log2(1 + SINR).
        No NOMA, SIC, or jammer interference.
        """
        gain = self.a2g_channel_gain(pos_user, pos_uav)
        return self.transmission_rate(
            tx_power, gain, bandwidth, noise_psd, interference_power=0.0)

    def sinr_under_jamming(self, tx_power, channel_gain_signal,
                           jammer_power, channel_gain_jammer,
                           noise_psd=NOISE_PSD, bandwidth=RBS_BANDWIDTH):
        """SINR = P·g_signal / (P_J·g_jammer + σ²)  (Paper §III.C Eq.)"""
        return (tx_power * channel_gain_signal /
                (jammer_power * channel_gain_jammer + noise_psd * bandwidth))


class NOMACommunication:
    """
    N-User SIC NOMA communication model (Paper §III.D).

    SIC decoding order π: ascending channel gain (weakest decoded first).
    SINR at decode position i (Paper Eq. 12):
      γ_{n,m} = p_n·g_{n,m} / (Σ_{r>i} p_{π(r)}·g_{π(r),m} + I_m + σ²)

    Power allocation (Paper Eq. 14): golden-section search over a boost
    parameter α that shifts power toward the weakest user, minimising
    total cluster transmission delay Σ_n D_n / R_n(p).
    """

    def __init__(self, channel_model=None):
        self.channel = channel_model or ChannelModel()

    def _compute_all_sinr_rates(self, power_alloc, gains_sorted, bandwidth, noise_psd):
        """Vectorised SIC rates for all N users — one NumPy pass."""
        N   = len(gains_sorted)
        bw  = bandwidth / max(N, 1)
        pg  = power_alloc * gains_sorted
        ici = np.cumsum(pg[::-1])[::-1] - pg
        return bw * np.log2(1.0 + pg / np.maximum(ici + noise_psd * bw, 1e-20))

    def compute_sinr_sic(self, user_idx, power_alloc, channel_gains, bandwidth, noise_psd=NOISE_PSD):
        """SINR at SIC decode position user_idx (Paper Eq. 12)."""
        p, g = np.asarray(power_alloc, float), np.asarray(channel_gains, float)
        bw   = bandwidth / max(len(g), 1)
        sig  = p[user_idx] * g[user_idx]
        ici  = np.dot(p[user_idx + 1:], g[user_idx + 1:])
        return sig / max(ici + noise_psd * bw, 1e-20)

    def compute_rate_sic(self, user_idx, power_alloc, channel_gains, bandwidth, noise_psd=NOISE_PSD):
        """Shannon rate for user at SIC position user_idx."""
        bw = bandwidth / max(len(channel_gains), 1)
        return bw * np.log2(1.0 + self.compute_sinr_sic(
            user_idx, power_alloc, channel_gains, bandwidth, noise_psd))

    def _cluster_delay(self, power_alloc, gains_sorted, sizes_sorted, bandwidth, noise_psd):
        rates = self._compute_all_sinr_rates(
            np.asarray(power_alloc), np.asarray(gains_sorted), bandwidth, noise_psd)
        return np.sum(np.asarray(sizes_sorted) / np.maximum(rates, 1e-10))

    def optimise_power_nuser(self, P_total, channel_gains, data_sizes,
                              bandwidth, noise_psd=NOISE_PSD, n_bisect=25):
        """
        N-user power optimisation via golden-section search (Paper Eq. 14).

        Parametrises all allocations with a single boost α ∈ [0, P_total/N]:
          p_0 = P_total/N + α         (weakest user, decoded first)
          p_i = P_total/N − α/(N−1)  (remaining users)

        Returns:
            power_alloc : np.array (W), aligned to ascending-gain SIC order
            order       : np.array mapping SIC order → original indices
        """
        N = len(channel_gains)
        if N == 0:
            return np.array([]), np.array([], dtype=int)
        if N == 1:
            return np.array([P_total]), np.array([0])

        order        = np.argsort(channel_gains)
        gains_sorted = np.array(channel_gains)[order]
        sizes_sorted = np.array(data_sizes)[order]
        eq_p         = P_total / N

        def build(alpha):
            p    = np.full(N, eq_p)
            p[0] = eq_p + alpha
            if N > 1:
                p[1:] = eq_p - alpha / (N - 1)
            p = np.clip(p, 0.0, P_total)
            return p / (p.sum() + 1e-20) * P_total

        lo, hi = 0.0, eq_p
        gr     = (np.sqrt(5) + 1) / 2
        for _ in range(n_bisect):
            if hi - lo < 1e-9:
                break
            c, d = hi - (hi - lo) / gr, lo + (hi - lo) / gr
            if (self._cluster_delay(build(c), gains_sorted, sizes_sorted, bandwidth, noise_psd) <
                    self._cluster_delay(build(d), gains_sorted, sizes_sorted, bandwidth, noise_psd)):
                hi = d
            else:
                lo = c

        return build((lo + hi) / 2), order

    def serve_cluster(self, user_indices, channel_gains, data_sizes,
                      P_total, bandwidth, noise_psd=NOISE_PSD):
        """
        Schedule all N devices in a UAV cluster using N-user SIC NOMA.

        Returns a list of dicts (in original user_indices order):
            user_idx : original index
            power    : allocated transmit power (W)
            rate     : achievable uplink rate (bps)
            tx_delay : data_size / rate (s)
        """
        N = len(user_indices)
        if N == 0:
            return []

        gains = np.array(channel_gains, dtype=float)
        sizes = np.array(data_sizes,    dtype=float)

        if N == 1:
            bw   = bandwidth
            sinr = P_total * gains[0] / max(NOISE_PSD * bw, 1e-20)
            rate = bw * np.log2(1.0 + sinr)
            return [{'user_idx': user_indices[0], 'power': P_total,
                     'rate': rate, 'tx_delay': sizes[0] / max(rate, 1e-10)}]

        alloc, order = self.optimise_power_nuser(
            P_total, gains.tolist(), sizes.tolist(), bandwidth, noise_psd)
        gains_sorted = gains[order]
        sizes_sorted = sizes[order]
        rates        = self._compute_all_sinr_rates(alloc, gains_sorted, bandwidth, noise_psd)

        results = [
            {'user_idx': user_indices[order[i]], 'power': float(alloc[i]),
             'rate': float(rates[i]),
             'tx_delay': float(sizes_sorted[i] / max(rates[i], 1e-10))}
            for i in range(N)
        ]
        return sorted(results, key=lambda d: d['user_idx'])

    def compute_pairing_index(self, channel_gain, file_size,
                               w_g=PAIRING_WEIGHT_CHANNEL, w_c=PAIRING_WEIGHT_FILE):
        """User pairing index for NOMA cluster formation (Paper §III.D)."""
        return (w_g * np.log(max(channel_gain, 1e-20)) -
                w_c * np.log(max(file_size, 1e-6)))
