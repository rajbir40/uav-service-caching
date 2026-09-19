"""
Cooperative Service Caching (3-Tier) — Paper §III.D and §IV.C
===============================================================
Tier 1 — Local UAV cache hit      → zero extra delay
Tier 2 — Cooperative peer fetch   → A2A transfer delay
Tier 3 — Base Station backhaul    → backhaul delay (no BS computation)

Ablation rows 5 and 6 (USE_COOP_CACHE=False):
  Tier 2 is skipped; cache misses fall directly to the BS.

Per-UAV placement uses the Frequency+Weight priority score (Paper §IV.C Eq.)
and a greedy knapsack under each UAV's capacity constraint.
Diversity-aware round-robin initialisation maximises collective coverage.
"""

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (
    NUM_SERVICE_PROGRAMS, PROGRAM_SIZES, CACHE_UPDATE_INTERVAL,
    OFFLOAD_DELAY_COOPERATIVE, OFFLOAD_DELAY_BS,
    CACHE_HIT_REWARD, CACHE_COOP_REWARD, CACHE_MISS_PENALTY,
    USE_COOP_CACHE,ZIPF_EXPONENT
)


class FrequencyWeightCaching:
    """
    Adaptive per-UAV caching using a Frequency+Weight priority score.

      S_l = λ_f · (N_l / Σ N_l') + λ_w · (W_l / Σ W_l')
            with λ_f = 0.6, λ_w = 0.4   (Paper §IV.C)
    """

    def __init__(self, num_programs, cache_capacity, program_sizes):
        self.num_programs   = num_programs
        self.cache_capacity = cache_capacity
        self.program_sizes  = np.array(program_sizes)
        self.task_counts    = np.zeros(num_programs)
        self.weight_scores  = np.zeros(num_programs)
        self.total_tasks    = 0

    def reset(self):
        self.task_counts[:]   = 0
        self.weight_scores[:] = 0
        self.total_tasks      = 0

    def update_statistics(self, program_id, completed=True):
        self.task_counts[program_id] += 1
        self.total_tasks             += 1
        if completed:
            self.weight_scores[program_id] += 1

    def compute_scores(self):
        P_needed = max(np.sum(self.task_counts > 0), 1)
        scores = np.zeros(self.num_programs)
        for p in range(self.num_programs):
            F_p = 0.6 * self.task_counts[p] / P_needed
            W_p = (0.4 * self.weight_scores[p] / self.total_tasks
                   if self.total_tasks > 0 else 0.0)
            scores[p] = F_p + W_p
        return scores

    def select_programs(self):
        """Greedy knapsack selection under storage capacity constraint."""
        scores       = self.compute_scores()
        sorted_progs = np.argsort(-scores)
        selected, used = [], 0.0

        for p in sorted_progs:
            if used + self.program_sizes[p] <= self.cache_capacity:
                selected.append(p)
                used += self.program_sizes[p]

        remaining = [p for p in range(self.num_programs) if p not in selected]
        np.random.shuffle(remaining)
        for p in remaining:
            if used + self.program_sizes[p] <= self.cache_capacity:
                selected.append(p)
                used += self.program_sizes[p]

        return selected


class CooperativeCacheManager:
    """
    3-tier cooperative cache manager for all UAVs (Paper §III.D, §IV.C).

    lookup() implements the tier resolution defined in Paper Eq. (fetch_delay):
      Tier 1: program in local cache         → delay = 0
      Tier 2: program in any peer UAV cache  → delay = OFFLOAD_DELAY_COOPERATIVE
      Tier 3: BS backhaul                    → delay = OFFLOAD_DELAY_BS

    update_caches() is called by env.step() at macro-interval boundaries.
    The calling frequency is controlled externally via CACHE_TIMESCALE_K_C.
    """

    def __init__(self, num_uavs, per_uav_capacities=None,
                 num_programs=NUM_SERVICE_PROGRAMS, seed=42):
        self.num_uavs    = num_uavs
        self.num_programs = num_programs
        self.rng         = np.random.RandomState(seed)

        self.per_uav_capacities = (
            list(per_uav_capacities) if per_uav_capacities is not None
            else [10] * num_uavs
        )

        self.program_sizes = self.rng.uniform(
            PROGRAM_SIZES[0], PROGRAM_SIZES[1], num_programs)

        ranks = np.arange(1, num_programs + 1, dtype=float)
        raw   = ranks ** (-ZIPF_EXPONENT)
        self.program_popularity = raw / raw.sum()

        self.mab_agents = [
            FrequencyWeightCaching(
                num_programs,
                self.per_uav_capacities[i],
                self.program_sizes,
            )
            for i in range(num_uavs)
        ]
        self.uav_caches: list[list[int]] = [[] for _ in range(num_uavs)]

        self._diversity_init()
        self._reset_stats()

    def _diversity_init(self):
        """Round-robin popularity-first placement to maximise collective coverage."""
        sorted_progs    = np.argsort(-self.program_popularity)
        self.uav_caches = [[] for _ in range(self.num_uavs)]
        uav_used        = [0.0] * self.num_uavs

        for prog_id in sorted_progs:
            best_uav, best_used = None, float('inf')
            for s in range(self.num_uavs):
                cap_left = self.per_uav_capacities[s] - uav_used[s]
                if self.program_sizes[prog_id] <= cap_left and uav_used[s] < best_used:
                    best_uav  = s
                    best_used = uav_used[s]
            if best_uav is not None:
                self.uav_caches[best_uav].append(prog_id)
                uav_used[best_uav] += self.program_sizes[prog_id]

    def _reset_stats(self):
        self.stats = {'local_hits': 0, 'coop_hits': 0, 'bs_fetches': 0,
                      'offload_links': []}

    def reset(self):
        self._diversity_init()
        for agent in self.mab_agents:
            agent.reset()
        self._reset_stats()

    def sample_required_program(self) -> int:
        return int(self.rng.choice(self.num_programs, p=self.program_popularity))

    def lookup(self, uav_id: int, required_program: int):
        """
        Resolve service availability across 3 tiers.

        Returns: (tier, delay, partner_id)
          tier       : 1 = local, 2 = cooperative A2A, 3 = BS backhaul
          delay      : additional latency (s)
          partner_id : cooperating UAV id (tier 2 only), else None
        """
        if required_program in self.uav_caches[uav_id]:
            self.stats['local_hits'] += 1
            return 1, 0.0, None

        if USE_COOP_CACHE:
            for other_id in range(self.num_uavs):
                if other_id != uav_id and required_program in self.uav_caches[other_id]:
                    self.stats['coop_hits'] += 1
                    self.stats['offload_links'].append((uav_id, other_id))
                    return 2, OFFLOAD_DELAY_COOPERATIVE, other_id

        self.stats['bs_fetches'] += 1
        return 3, OFFLOAD_DELAY_BS, None

    def update_caches(self, slot: int):
        """Refresh all UAV caches using current MAB priority scores."""
        if slot == 0:
            return
        for s in range(self.num_uavs):
            self.uav_caches[s] = self.mab_agents[s].select_programs()

    def log_task(self, uav_id: int, program_id: int, completed: bool = True):
        self.mab_agents[uav_id].update_statistics(program_id, completed)

    def get_cache_state(self, uav_id: int) -> np.ndarray:
        state = np.zeros(self.num_programs, dtype=np.float32)
        for p in self.uav_caches[uav_id]:
            state[p] = 1.0
        return state

    def get_collective_coverage(self) -> float:
        all_cached = set()
        for cache in self.uav_caches:
            all_cached.update(cache)
        return len(all_cached) / max(self.num_programs, 1)

    def clear_slot_links(self):
        self.stats['offload_links'] = []
