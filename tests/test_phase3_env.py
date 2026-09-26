#!/usr/bin/env python3
"""
Phase 3 Environment Tests
==========================

Tests for atomic service caching and cache-hit/miss behavior.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, NUM_SERVICES, SERVICE_SIZE_RANGE, PHASE1_UAV_BATTERY_J


class TestPhase3Caching(unittest.TestCase):
    """Test Phase 3 caching and service replication."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        self.env.service_sizes = {f"service_{i}": SERVICE_SIZE_RANGE[1] for i in range(1, NUM_SERVICES + 1)}

    def test_service_catalog(self):
        """Test that the service catalog is correctly initialized."""
        self.assertEqual(len(self.env.service_sizes), NUM_SERVICES)
        for service in self.env.service_sizes:
            self.assertTrue(SERVICE_SIZE_RANGE[0] <= self.env.service_sizes[service] <= SERVICE_SIZE_RANGE[1])

    def test_cache_capacity(self):
        """Test that cache capacity is enforced."""
        obs, state = self.env.reset()

        # Assign a UAV with a small cache
        uav = self.env.uavs[0]
        uav.cache_capacity = 100.0

        # Insert services until cache is full
        for service in self.env.service_sizes:
            self.env.uavs[0].insert_service(service)
            self.assertTrue(service in uav.service_cache)
            self.assertEqual(len(uav.service_cache), len(self.env.service_sizes.keys()))

        # Try to insert another service (should fail due to capacity)
        self.assertFalse(self.env.uavs[0].insert_service("service_11"))
        self.assertEqual(len(uav.service_cache), len(self.env.service_sizes.keys()))

    def test_service_insertion(self):
        """Test service insertion and eviction."""
        obs, state = self.env.reset()

        uav = self.env.uavs[0]
        uav.cache_capacity = 100.0

        # Insert services
        self.assertTrue(uav.insert_service("service_1"))
        self.assertTrue(uav.insert_service("service_2"))
        self.assertFalse(uav.insert_service("service_1"))  # Already exists

        # Evict a service
        self.assertTrue(uav.evict_service("service_1"))
        self.assertFalse("service_1" in uav.service_cache)

    def test_local_hit_detection(self):
        """Test local hit detection for tasks."""
        obs, state = self.env.reset()

        # Insert a service into UAV 0
        self.env.uavs[0].insert_service("service_1")

        # Generate a task
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        for task in tasks:
            # Check if the task is a local hit
            self.assertIsNotNone(task.local_hit, msg="Task should have local_hit flag")
            # Ensure the task is assigned to a UAV with the service
            uav = self.env.uavs[task.assigned_uav_id]
            self.assertTrue("service_1" in uav.service_cache, msg="Service should be cached")
            self.assertTrue(task.local_hit, msg="Task should be a local hit")

    def test_cache_miss_latency(self):
        """Test latency calculation for cache misses."""
        obs, state = self.env.reset()

        # Insert a service into UAV 0
        self.env.uavs[0].insert_service("service_1")

        # Generate a task that is not a local hit
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        for task in tasks:
            if not task.local_hit:
                self.assertGreater(task.t_acq, 0.0, msg="Cache miss latency should be positive")

    def test_task_latency_components(self):
        """Test that latency components are correctly tracked."""
        obs, state = self.env.reset()

        # Insert a service into UAV 0
        self.env.uavs[0].insert_service("service_1")

        # Simulate task processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        for task in tasks:
            self.assertGreaterEqual(task.t_upload, 0.0, msg="Upload latency should be non-negative")
            self.assertGreaterEqual(task.t_queue, 0.0, msg="Queue latency should be non-negative")
            self.assertGreaterEqual(task.t_compute, 0.0, msg="Compute latency should be non-negative")
            self.assertGreaterEqual(task.t_total, 0.0, msg="Total latency should be non-negative")

    def test_cache_metrics(self):
        """Test cache hit/miss metrics."""
        obs, state = self.env.reset()

        # Insert a service into UAV 0
        self.env.uavs[0].insert_service("service_1")

        # Simulate task generation and processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check cache metrics
        cache_metrics = self.env.info["cache_local_hits"]
        cache_misses = self.env.info["cache_misses"]
        self.assertGreater(cache_metrics, 0, msg="Local hits should be recorded")
        self.assertGreater(cache_misses, 0, msg="Cache misses should be recorded")

    def test_deterministic_behavior(self):
        """Test deterministic behavior with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        # Insert services into UAVs
        env1.uavs[0].insert_service("service_1")
        env2.uavs[0].insert_service("service_1")

        # Simulate task processing
        for _ in range(5):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        # Compare completed tasks
        tasks1 = env1.completed_tasks
        tasks2 = env2.completed_tasks
        self.assertEqual(len(tasks1), len(tasks2), msg="Number of completed tasks should match")
        for t1, t2 in zip(tasks1, tasks2):
            self.assertEqual(t1.task_id, t2.task_id, msg="Task IDs should match")
            self.assertEqual(t1.local_hit, t2.local_hit, msg="Local hit status should match")


if __name__ == "__main__":
    unittest.main()