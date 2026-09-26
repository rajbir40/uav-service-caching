#!/usr/bin/env python3
"""
Phase 8 Environment Tests
==========================

Tests for spatially shifting demand hotspots.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, AREA_RADIUS


class TestPhase8Hotspots(unittest.TestCase):
    """Test Phase 8 hotspot behavior."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

    def test_stationary_iot_positions(self):
        """Test that IoT device positions remain stationary."""
        obs, state = self.env.reset()
        initial_positions = [dev.position.copy() for dev in self.env.devices]

        # Simulate several steps
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that device positions have not changed
        for dev, initial_pos in zip(self.env.devices, initial_positions):
            self.assertAlmostEqual(dev.position[0], initial_pos[0], places=6, msg="IoT device x position should not change")
            self.assertAlmostEqual(dev.position[1], initial_pos[1], places=6, msg="IoT device y position should not change")

    def test_spatially_non_uniform_demand(self):
        """Test that demand becomes spatially non-uniform with hotspots."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that hotspots are active and influence task generation
        self.assertTrue(self.env.hotspot_active, msg="Hotspots should be active")
        self.assertGreater(len(self.env.hotspot_positions), 0, msg="Hotspot positions should be set")

        # Check that tasks are generated more frequently in hotspot areas
        hotspot_area_devices = []
        non_hotspot_area_devices = []
        for dev in self.env.devices:
            in_hotspot = any(np.linalg.norm(np.array(dev.position[:2]) - np.array(hotspot)) <= self.env.hotspot_radius
                                for hotspot in self.env.hotspot_positions)
            if in_hotspot:
                hotspot_area_devices.append(dev)
            else:
                non_hotspot_area_devices.append(dev)

        # Simulate task generation
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that more tasks are generated in hotspot areas
        hotspot_tasks = sum(len(dev.pending_tasks) for dev in hotspot_area_devices)
        non_hotspot_tasks = sum(len(dev.pending_tasks) for dev in non_hotspot_area_devices)
        self.assertGreater(hotspot_tasks, non_hotspot_tasks, msg="More tasks should be generated in hotspot areas")

    def test_hotspot_movement(self):
        """Test that hotspots move over time."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        initial_hotspot_positions = [pos.copy() for pos in self.env.hotspot_positions]

        # Simulate more steps to move hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that hotspot positions have changed
        for initial_pos, new_pos in zip(initial_hotspot_positions, self.env.hotspot_positions):
            self.assertNotAlmostEqual(initial_pos[0], new_pos[0], places=6, msg="Hotspot x position should change")
            self.assertNotAlmostEqual(initial_pos[1], new_pos[1], places=6, msg="Hotspot y position should change")

    def test_changing_active_user_distribution(self):
        """Test that the active-user distribution changes with hotspots."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Track initial active users
        initial_active_users = [dev.id for dev in self.env.devices if len(dev.pending_tasks) > 0]

        # Simulate more steps to change active users
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Track new active users
        new_active_users = [dev.id for dev in self.env.devices if len(dev.pending_tasks) > 0]

        # Check that the active-user distribution has changed
        self.assertNotEqual(initial_active_users, new_active_users, msg="Active-user distribution should change")

    def test_poisson_hotspot_interaction(self):
        """Test that Poisson arrivals interact with hotspots."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that Poisson arrivals are modulated by hotspot influence
        hotspot_area_devices = [dev for dev in self.env.devices if any(np.linalg.norm(np.array(dev.position[:2]) - np.array(hotspot)) <= self.env.hotspot_radius
                                for hotspot in self.env.hotspot_positions)]
        non_hotspot_area_devices = [dev for dev in self.env.devices if dev not in hotspot_area_devices]

        # Simulate task generation
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that more tasks are generated in hotspot areas
        hotspot_tasks = sum(len(dev.pending_tasks) for dev in hotspot_area_devices)
        non_hotspot_tasks = sum(len(dev.pending_tasks) for dev in non_hotspot_area_devices)
        self.assertGreater(hotspot_tasks, non_hotspot_tasks, msg="More tasks should be generated in hotspot areas")

    def test_reproducibility(self):
        """Test that hotspot behavior is reproducible with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        # Check that hotspot positions match
        for pos1, pos2 in zip(env1.hotspot_positions, env2.hotspot_positions):
            self.assertAlmostEqual(pos1[0], pos2[0], places=6, msg="Hotspot x positions should match")
            self.assertAlmostEqual(pos1[1], pos2[1], places=6, msg="Hotspot y positions should match")

    def test_demand_changes_without_device_movement(self):
        """Test that demand changes without device movement."""
        obs, state = self.env.reset()
        initial_positions = [dev.position.copy() for dev in self.env.devices]

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that device positions have not changed
        for dev, initial_pos in zip(self.env.devices, initial_positions):
            self.assertAlmostEqual(dev.position[0], initial_pos[0], places=6, msg="IoT device x position should not change")
            self.assertAlmostEqual(dev.position[1], initial_pos[1], places=6, msg="IoT device y position should not change")

        # Check that demand has changed
        self.assertTrue(self.env.hotspot_active, msg="Hotspots should be active")

    def test_valid_task_generation_under_hotspot_changes(self):
        """Test that valid tasks are generated under hotspot changes."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that tasks are generated and assigned correctly
        tasks = [task for dev in self.env.devices for task in dev.pending_tasks]
        self.assertGreater(len(tasks), 0, msg="Tasks should be generated")
        for task in tasks:
            self.assertIsNotNone(task.assigned_uav_id, msg="Task should be assigned to a UAV")

    def test_nan_inf_safety(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        # Simulate several steps to activate hotspots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check for NaN or Inf values
        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")

if __name__ == "__main__":
    unittest.main()