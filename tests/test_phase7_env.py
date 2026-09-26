#!/usr/bin/env python3
"""
Phase 7 Environment Tests
==========================

Tests for CVaR/P95/P99 over the active population.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS


class TestPhase7TailMetrics(unittest.TestCase):
    """Test Phase 7 tail metrics."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

    def test_average_latency(self):
        """Test average latency calculation."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info["avg_latency"], 0.0, msg="Average latency should be non-negative")

    def test_p95_latency(self):
        """Test P95 latency calculation."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info["p95_latency"], 0.0, msg="P95 latency should be non-negative")

    def test_p99_latency(self):
        """Test P99 latency calculation."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info["p99_latency"], 0.0, msg="P99 latency should be non-negative")

    def test_cvar_latency(self):
        """Test CVaR latency calculation."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info["cvar_latency"], 0.0, msg="CVaR latency should be non-negative")

    def test_correct_ordering(self):
        """Test that P95 <= P99 <= CVaR."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertLessEqual(info["p95_latency"], info["p99_latency"], msg="P95 should be less than or equal to P99")
        self.assertLessEqual(info["p99_latency"], info["cvar_latency"], msg="P99 should be less than or equal to CVaR")

    def test_device_level_aggregation(self):
        """Test device-level latency aggregation."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertIn("device_avg_latencies", info, msg="Device average latencies should be in info")
        self.assertIn("device_p95_latencies", info, msg="Device P95 latencies should be in info")
        self.assertIn("device_p99_latencies", info, msg="Device P99 latencies should be in info")
        self.assertIn("device_cvar_latencies", info, msg="Device CVaR latencies should be in info")

    def test_small_populations(self):
        """Test handling of small/empty populations."""
        obs, state = self.env.reset()

        # Simulate task processing with few tasks
        for _ in range(2):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info["avg_latency"], 0.0, msg="Average latency should be non-negative")
        self.assertGreaterEqual(info["p95_latency"], 0.0, msg="P95 latency should be non-negative")
        self.assertGreaterEqual(info["p99_latency"], 0.0, msg="P99 latency should be non-negative")
        self.assertGreaterEqual(info["cvar_latency"], 0.0, msg="CVaR latency should be non-negative")

    def test_deterministic_results(self):
        """Test deterministic results with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        # Simulate task processing
        for _ in range(5):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        info1 = env1.info
        info2 = env2.info

        self.assertAlmostEqual(info1["avg_latency"], info2["avg_latency"], places=6, msg="Average latencies should match")
        self.assertAlmostEqual(info1["p95_latency"], info2["p95_latency"], places=6, msg="P95 latencies should match")
        self.assertAlmostEqual(info1["p99_latency"], info2["p99_latency"], places=6, msg="P99 latencies should match")
        self.assertAlmostEqual(info1["cvar_latency"], info2["cvar_latency"], places=6, msg="CVaR latencies should match")

    def test_nan_inf_safety(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertFalse(np.isnan(info["avg_latency"]), msg="Average latency should not be NaN")
        self.assertFalse(np.isinf(info["avg_latency"]), msg="Average latency should not be Inf")
        self.assertFalse(np.isnan(info["p95_latency"]), msg="P95 latency should not be NaN")
        self.assertFalse(np.isinf(info["p95_latency"]), msg="P95 latency should not be Inf")
        self.assertFalse(np.isnan(info["p99_latency"]), msg="P99 latency should not be NaN")
        self.assertFalse(np.isinf(info["p99_latency"]), msg="P99 latency should not be Inf")
        self.assertFalse(np.isnan(info["cvar_latency"]), msg="CVaR latency should not be NaN")
        self.assertFalse(np.isinf(info["cvar_latency"]), msg="CVaR latency should not be Inf")

    def test_metric_exposure(self):
        """Test that metrics are exposed through the environment."""
        obs, state = self.env.reset()

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertIn("avg_latency", info, msg="Average latency should be in info")
        self.assertIn("p95_latency", info, msg="P95 latency should be in info")
        self.assertIn("p99_latency", info, msg="P99 latency should be in info")
        self.assertIn("cvar_latency", info, msg="CVaR latency should be in info")

if __name__ == "__main__":
    unittest.main()