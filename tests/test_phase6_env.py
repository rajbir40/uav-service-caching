#!/usr/bin/env python3
"""
Phase 6 Environment Tests
==========================

Tests for continuous 2-D UAV movement with bounded speed, area, and collision constraints.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, AREA_RADIUS, MIN_UAV_SEPARATION, UAV_PROPULSION_C1, UAV_PROPULSION_C2


class TestPhase6Trajectory(unittest.TestCase):
    """Test Phase 6 trajectory model."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

    def test_2d_uav_movement(self):
        """Test continuous 2-D UAV movement."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        initial_position = uav.position.copy()

        # Apply movement action
        actions = np.array([0.5, 0.5, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
        self.env.step(actions)

        # Check that position has changed
        self.assertNotEqual(uav.position[0], initial_position[0], msg="UAV x position should change")
        self.assertNotEqual(uav.position[1], initial_position[1], msg="UAV y position should change")

    def test_speed_bound(self):
        """Test that UAV speed is bounded."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        max_speed = uav.max_displacement

        # Apply maximum speed action
        actions = np.array([1.0, 1.0, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
        self.env.step(actions)

        # Calculate actual speed
        speed = np.hypot(uav.vx, uav.vy)
        self.assertLessEqual(speed, max_speed, msg="UAV speed should not exceed max speed")

    def test_area_boundary(self):
        """Test that UAVs stay within the bounded operating region."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        initial_position = uav.position.copy()

        # Move UAV to the boundary
        actions = np.array([1.0, 0.0, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
        self.env.step(actions)

        # Check that UAV stays within the area
        distance_from_center = np.linalg.norm(uav.position[:2])
        self.assertLessEqual(distance_from_center, AREA_RADIUS, msg="UAV should stay within the area boundary")

    def test_collision_separation(self):
        """Test collision/separation constraint between UAVs."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]

        # Position UAVs close to each other
        uav0.position[0] = 10.0
        uav0.position[1] = 10.0
        uav1.position[0] = 10.0 + MIN_UAV_SEPARATION / 2
        uav1.position[1] = 10.0

        # Apply movement actions to bring them closer
        actions = np.array([1.0, 0.0, 0.0, -1.0, 0.0, 0.0] + [0.0] * (NUM_UAVS * 5 - 6))
        self.env.step(actions)

        # Check that UAVs maintain minimum separation
        distance = np.linalg.norm(uav0.position[:2] - uav1.position[:2])
        self.assertGreaterEqual(distance, MIN_UAV_SEPARATION, msg="UAVs should maintain minimum separation")

    def test_distance_dependent_a2g_behavior(self):
        """Test that A2G distances affect communication rates."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        device = self.env.devices[0]

        # Initial distance and rate
        initial_distance = np.linalg.norm(uav.position[:2] - device.position[:2])
        initial_rate = self.env.calculate_a2g_rate(device, uav)

        # Move UAV closer to the device
        uav.position[0] = device.position[0]
        uav.position[1] = device.position[1]
        new_rate = self.env.calculate_a2g_rate(device, uav)

        # Rate should increase as distance decreases
        self.assertGreater(new_rate, initial_rate, msg="A2G rate should increase as distance decreases")

    def test_distance_dependent_a2a_behavior(self):
        """Test that A2A distances affect communication rates."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]

        # Initial distance and rate
        initial_distance = np.linalg.norm(uav0.position[:2] - uav1.position[:2])
        initial_rate = self.env.calculate_a2a_rate(uav0, uav1)

        # Move UAVs closer to each other
        uav0.position[0] = uav1.position[0]
        uav0.position[1] = uav1.position[1]
        new_rate = self.env.calculate_a2a_rate(uav0, uav1)

        # Rate should increase as distance decreases
        self.assertGreater(new_rate, initial_rate, msg="A2A rate should increase as distance decreases")

    def test_propulsion_energy_consistency(self):
        """Test that propulsion energy uses actual movement."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        initial_energy = uav.propulsion_energy

        # Apply movement action
        actions = np.array([0.5, 0.5, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
        self.env.step(actions)

        # Calculate expected propulsion energy
        speed = np.hypot(uav.vx, uav.vy)
        expected_energy = UAV_PROPULSION_C1 + UAV_PROPULSION_C2 * speed**2

        # Propulsion energy should increase with movement
        self.assertGreater(uav.propulsion_energy, initial_energy, msg="Propulsion energy should increase with movement")

    def test_deterministic_behavior(self):
        """Test deterministic behavior with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        # Apply the same actions
        actions = np.array([0.5, 0.5, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
        env1.step(actions)
        env2.step(actions)

        # Check that UAV positions match
        for uav1, uav2 in zip(env1.uavs, env2.uavs):
            self.assertAlmostEqual(uav1.position[0], uav2.position[0], places=6, msg="UAV x positions should match")
            self.assertAlmostEqual(uav1.position[1], uav2.position[1], places=6, msg="UAV y positions should match")

    def test_nan_inf_safety(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        # Apply several movement actions
        for _ in range(5):
            actions = np.array([0.5, 0.5, 0.0] + [0.0] * (NUM_UAVS * 5 - 3))
            self.env.step(actions)

        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")

if __name__ == "__main__":
    unittest.main()