#!/usr/bin/env python3
"""
Phase 1 Environment Tests
==========================

Tests for the basic environment setup with stationary IoT devices, fixed UAV positions, and no caching/movement.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, PHASE1_UAV_BATTERY_J, PHASE1_LAT_NORM, PHASE1_ENERGY_NORM


class TestPhase1Environment(unittest.TestCase):
    """Test Phase 1 environment initialization and task processing."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS)

    def test_environment_initialization(self):
        """Test that the environment initializes with stationary devices and UAVs."""
        obs, state = self.env.reset()

        # Check that UAVs are stationary
        for uav_id in range(NUM_UAVS):
            uav_obs = obs[uav_id]
            self.assertAlmostEqual(uav_obs[3], 0.0, places=6, msg="UAV x velocity should be zero")
            self.assertAlmostEqual(uav_obs[4], 0.0, places=6, msg="UAV y velocity should be zero")
            self.assertAlmostEqual(uav_obs[5], 0.0, places=6, msg="UAV battery energy should be full")

    def test_task_generation(self):
        """Test task generation and assignment to UAVs."""
        obs, state = self.env.reset()

        # Generate tasks
        for _ in range(10):
            self.env.step([0.0, 0.0, 0.0] * NUM_UAVS)

        # Check that tasks are generated and assigned
        tasks_pending = self.env.info["tasks_pending"]
        self.assertGreater(tasks_pending, 0, msg="Tasks should be generated and pending")

    def test_latency_calculation(self):
        """Test that latency calculations are correct."""
        obs, state = self.env.reset()

        # Simulate task upload, queue, and compute
        task = self.env.devices[0].pending_tasks[0]
        task.t_upload = 0.5
        task.t_queue = 0.3
        task.t_compute = 0.4
        task.t_total = 1.2

        # Check total latency calculation
        total_latency = self.env.calculate_total_latency(task.t_upload, task.t_queue, task.t_compute)
        self.assertAlmostEqual(total_latency, 1.2, places=6, msg="Total latency should match sum of components")

    def test_energy_budget(self):
        """Test energy budget and penalties."""
        obs, state = self.env.reset()

        # Simulate energy usage
        for uav_id in range(NUM_UAVS):
            self.env.uavs[uav_id].total_energy = PHASE1_UAV_BATTERY_J
            self.env.uavs[uav_id].battery_energy = PHASE1_UAV_BATTERY_J

        # Simulate step to use energy
        self.env.step([0.0] * NUM_UAVS * 5)

        # Check energy usage and penalty
        energy_used = sum(uav.total_energy for uav in self.env.uavs)
        self.assertLess(energy_used, PHASE1_UAV_BATTERY_J * NUM_UAVS, msg="Energy should not exceed budget")

    def test_no_caching_influence(self):
        """Test that tasks are processed without caching influence."""
        obs, state = self.env.reset()

        # Generate tasks
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that tasks are assigned to the nearest UAV
        for task in self.env.devices[0].pending_tasks:
            self.assertIsNotNone(task.assigned_uav_id, msg="Task should be assigned to a UAV")
            # Ensure tasks are always processed locally
            self.assertEqual(task.assigned_uav_id, self.env.uavs[0].id, msg="Task should be assigned to the first UAV")

    def test_no_uav_movement(self):
        """Test that UAVs do not move."""
        obs, state = self.env.reset()

        # Check initial positions
        initial_positions = [uav.position.copy() for uav in self.env.uavs]

        # Simulate steps
        for _ in range(10):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that positions remain unchanged
        for i, uav in enumerate(self.env.uavs):
            self.assertTrue(np.allclose(initial_positions[i], uav.position, atol=1e-6),
                            msg=f"UAV {i} should not move")

    def test_no_nan_inf(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        # Generate tasks and process them
        for _ in range(10):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check for NaN or Inf in state and observations
        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")
        self.assertFalse(np.isnan(state).any(), msg="No NaN values in state")
        self.assertFalse(np.isinf(state).any(), msg="No Inf values in state")


if __name__ == "__main__":
    unittest.main()