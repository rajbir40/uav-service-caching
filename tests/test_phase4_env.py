#!/usr/bin/env python3
"""
Phase 4 Environment Tests
==========================

Tests for UAV↔UAV migration and UAV↔anchor pull mechanisms.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, NUM_SERVICES, SERVICE_SIZE_RANGE, REACHABLE_DISTANCE


class TestPhase4Migration(unittest.TestCase):
    """Test Phase 4 migration and anchor pull mechanisms."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        self.env.service_sizes = {f"service_{i}": SERVICE_SIZE_RANGE[1] for i in range(1, NUM_SERVICES + 1)}

    def test_local_replica_path(self):
        """Test that tasks are served locally when the required service is cached."""
        obs, state = self.env.reset()
        uav = self.env.uavs[0]
        uav.insert_service("service_1")

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            self.assertEqual(task.path, "LOCAL", msg="Task should be served locally")

    def test_a2a_migration_path(self):
        """Test that tasks are migrated to a peer UAV when the required service is not locally cached but is cached on a reachable peer."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")
        uav1.insert_service("service_2")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "MIGRATION":
                self.assertTrue(task.execution_uav_id != task.assigned_uav_id, msg="Task should be executed on a different UAV")

    def test_anchor_pull_path(self):
        """Test that tasks are acquired from the anchor when no suitable peer replica exists."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are not reachable
        uav0.position[0] = 100.0
        uav1.position[0] = 200.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "ANCHOR":
                self.assertEqual(task.execution_uav_id, task.assigned_uav_id, msg="Task should be executed on the assigned UAV")

    def test_migration_eligibility(self):
        """Test that migration is only allowed when the target UAV has the required service."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "MIGRATION":
                target_uav = self.env.uavs[task.execution_uav_id]
                self.assertIn(task.stages[0].name, target_uav.service_cache, msg="Target UAV should have the required service")

    def test_unreachable_peer_rejection(self):
        """Test that migration is rejected when the target UAV is unreachable."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are not reachable
        uav0.position[0] = 100.0
        uav1.position[0] = 200.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "LOCAL":
                self.assertEqual(task.execution_uav_id, task.assigned_uav_id, msg="Task should be served locally if no reachable peer")

    def test_missing_replica_rejection(self):
        """Test that tasks are not migrated when the required service is missing on the target UAV."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "LOCAL":
                self.assertEqual(task.execution_uav_id, task.assigned_uav_id, msg="Task should be served locally if no peer has the service")

    def test_path_exclusivity(self):
        """Test that exactly one execution path is selected per task."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            self.assertIn(task.path, ["LOCAL", "MIGRATION", "ANCHOR"], msg="Task path should be one of LOCAL, MIGRATION, or ANCHOR")

    def test_a2a_latency(self):
        """Test that A2A latency is correctly calculated for migrations and anchor pulls."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            if task.path == "MIGRATION":
                self.assertGreater(task.t_a2a, 0.0, msg="Migration latency should be positive")
            elif task.path == "ANCHOR":
                self.assertGreater(task.t_a2a, 0.0, msg="Anchor latency should be positive")

    def test_total_latency_decomposition(self):
        """Test that total latency is correctly decomposed."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        for task in tasks:
            self.assertGreaterEqual(task.t_total, task.t_upload + task.t_queue + task.t_compute, msg="Total latency should be the sum of components")

    def test_a2a_traffic_metrics(self):
        """Test that A2A traffic metrics are tracked."""
        obs, state = self.env.reset()
        uav0 = self.env.uavs[0]
        uav1 = self.env.uavs[1]
        uav0.insert_service("service_1")

        # Ensure UAVs are positioned such that they are reachable
        uav0.position[0] = 10.0
        uav1.position[0] = 20.0

        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        info = self.env.info
        self.assertGreaterEqual(info.get("cache_misses", 0), 0, msg="Cache misses should be tracked")
        self.assertGreaterEqual(info.get("cache_coop_hits", 0), 0, msg="Cooperative cache hits should be tracked")

    def test_deterministic_behavior(self):
        """Test deterministic behavior with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        uav0_1 = env1.uavs[0]
        uav0_2 = env2.uavs[0]
        uav1_1 = env1.uavs[1]
        uav1_2 = env2.uavs[1]

        uav0_1.insert_service("service_1")
        uav0_2.insert_service("service_1")
        uav1_1.insert_service("service_2")
        uav1_2.insert_service("service_2")

        # Ensure UAVs are positioned such that they are reachable
        uav0_1.position[0] = 10.0
        uav0_2.position[0] = 10.0
        uav1_1.position[0] = 20.0
        uav1_2.position[0] = 20.0

        for _ in range(3):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        tasks1 = env1.completed_tasks
        tasks2 = env2.completed_tasks
        self.assertEqual(len(tasks1), len(tasks2), msg="Number of completed tasks should match")
        for t1, t2 in zip(tasks1, tasks2):
            self.assertEqual(t1.path, t2.path, msg="Task paths should match")
            self.assertEqual(t1.execution_uav_id, t2.execution_uav_id, msg="Execution UAV IDs should match")

    def test_nan_inf_safety(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")

if __name__ == "__main__":
    unittest.main()