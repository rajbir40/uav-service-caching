#!/usr/bin/env python3
"""
Phase 5 Environment Tests
==========================

Tests for bounded energy budget and energy penalties.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, PHASE1_UAV_BATTERY_J, UAV_PROPULSION_C1, UAV_PROPULSION_C2, UAV_TX_POWER, UAV_RX_POWER, COMPUTATION_ENERGY_CONSTANT


class TestPhase5Energy(unittest.TestCase):
    """Test Phase 5 energy model and penalties."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

    def test_propulsion_energy(self):
        """Test propulsion energy calculation."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.vx = 10.0
        uav.vy = 10.0
        speed = np.hypot(uav.vx, uav.vy)
        expected_energy = UAV_PROPULSION_C1 + UAV_PROPULSION_C2 * speed**2
        self.env._move_uavs([0.0] * NUM_UAVS * 5, np.zeros(NUM_UAVS))
        self.assertAlmostEqual(uav.propulsion_energy, expected_energy, places=6, msg="Propulsion energy should match expected value")

    def test_communication_energy(self):
        """Test communication energy calculation."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.communication_energy = 0.0
        self.env._process_uploads(0.0, np.zeros(NUM_UAVS))
        self.assertGreater(uav.communication_energy, 0.0, msg="Communication energy should be positive")

    def test_computation_energy(self):
        """Test computation energy calculation."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.computation_energy = 0.0
        self.env._process_compute(0.0, np.zeros(NUM_UAVS), np.zeros(NUM_UAVS))
        self.assertGreater(uav.computation_energy, 0.0, msg="Computation energy should be positive")

    def test_battery_depletion(self):
        """Test battery depletion and energy never becomes negative."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.battery_energy = 10.0
        uav.propulsion_energy = 15.0
        self.env._move_uavs([0.0] * NUM_UAVS * 5, np.zeros(NUM_UAVS))
        self.assertEqual(uav.battery_energy, 0.0, msg="Battery energy should not go negative")

    def test_zero_energy_uav_behavior(self):
        """Test zero-energy UAV behavior."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.battery_energy = 0.0
        self.env._move_uavs([0.0] * NUM_UAVS * 5, np.zeros(NUM_UAVS))
        self.assertEqual(uav.vx, 0.0, msg="UAV should not move when battery is zero")
        self.assertEqual(uav.vy, 0.0, msg="UAV should not move when battery is zero")

    def test_a2a_energy(self):
        """Test A2A energy calculation for migrations and anchor pulls."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0
        uav0.insert_service("service_1")
        uav1.insert_service("service_2")

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "MIGRATION":
                self.assertGreater(task.migration_energy, 0.0, msg="Migration energy should be positive")
            elif task.path == "ANCHOR":
                self.assertGreater(task.anchor_energy, 0.0, msg="Anchor energy should be positive")

    def test_fleet_energy_accounting(self):
        """Test fleet energy accounting."""
        obs, state = self.env.reset()
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        total_energy = sum(uav.total_energy_consumed for uav in self.env.uavs)
        self.assertGreater(total_energy, 0.0, msg="Total energy consumed should be positive")

    def test_deterministic_behavior(self):
        """Test deterministic behavior with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        for _ in range(3):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        for uav1, uav2 in zip(env1.uavs, env2.uavs):
            self.assertEqual(uav1.battery_energy, uav2.battery_energy, msg="Battery energy should match")
            self.assertEqual(uav1.total_energy_consumed, uav2.total_energy_consumed, msg="Total energy consumed should match")

    def test_nan_inf_safety(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")

if __name__ == "__main__":
    unittest.main()