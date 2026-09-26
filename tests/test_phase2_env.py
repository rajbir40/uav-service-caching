#!/usr/bin/env python3
"""
Phase 2 Environment Tests
==========================

Tests for dynamic active/inactive task generation using Poisson arrivals.
"""

import unittest
import numpy as np
from src.env import MultiUAVMECEnv
from config import NUM_USERS, NUM_UAVS, TASK_GENERATION_PROB, PHASE1_UAV_BATTERY_J


class TestPhase2Environment(unittest.TestCase):
    """Test Phase 2 environment with Poisson arrivals and active/inactive users."""

    def setUp(self):
        self.env = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS)

    def test_poisson_arrivals(self):
        """Test that tasks are generated according to Poisson arrivals."""
        obs, state = self.env.reset()

        # Simulate a few slots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks_pending = self.env.info["tasks_pending"]
        self.assertGreater(tasks_pending, 0, msg="Tasks should be generated according to Poisson arrivals")

    def test_active_inactive_users(self):
        """Test active/inactive toggling of users."""
        obs, state = self.env.reset()

        # Check initial state
        for dev in self.env.devices:
            self.assertEqual(len(dev.pending_tasks), 0, msg="No pending tasks initially")

        # Simulate task generation
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Check that tasks are added to pending tasks
        for dev in self.env.devices:
            self.assertGreater(len(dev.pending_tasks), 0, msg="Device should have pending tasks")

        # Simulate processing of tasks
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        # Ensure tasks are processed and removed from pending
        for dev in self.env.devices:
            self.assertEqual(len(dev.pending_tasks), 0, msg="Pending tasks should be processed")

    def test_task_id_and_timestamp(self):
        """Test that tasks have unique IDs and timestamps."""
        obs, state = self.env.reset()

        # Generate tasks
        for _ in range(3):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        # Check task IDs are unique
        task_ids = [task.task_id for task in tasks]
        self.assertEqual(len(task_ids), len(set(task_ids)), msg="Task IDs should be unique")

        # Check timestamps are set
        for task in tasks:
            self.assertIsNotNone(task.timestamp, msg="Task timestamp should be set")

    def test_fifo_queue(self):
        """Test that tasks are processed in FIFO order."""
        obs, state = self.env.reset()

        # Generate tasks
        for _ in range(2):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        # Check FIFO order
        for i, task in enumerate(tasks):
            self.assertEqual(task.task_id, i + 1, msg=f"Task {i + 1} should have ID {i + 1}")

    def test_latency_decomposition(self):
        """Test latency decomposition for tasks spanning multiple slots."""
        obs, state = self.env.reset()

        # Simulate task processing over multiple slots
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        for task in tasks:
            # Check latency components
            self.assertGreaterEqual(task.t_upload, 0.0, msg="Upload latency should be non-negative")
            self.assertGreaterEqual(task.t_queue, 0.0, msg="Queue latency should be non-negative")
            self.assertGreaterEqual(task.t_compute, 0.0, msg="Compute latency should be non-negative")
            self.assertGreaterEqual(task.t_total, 0.0, msg="Total latency should be non-negative")

    def test_deadline_tracking(self):
        """Test deadline tracking for tasks."""
        obs, state = self.env.reset()

        # Simulate task generation and processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        tasks = self.env.completed_tasks
        self.assertGreater(len(tasks), 0, msg="No tasks completed")

        for task in tasks:
            self.assertIsInstance(task.deadline_missed, bool, msg="Deadline_missed should be boolean")

    def test_no_nan_inf(self):
        """Test that no NaN or Inf values are introduced."""
        obs, state = self.env.reset()

        # Simulate task generation and processing
        for _ in range(5):
            self.env.step([0.0] * NUM_UAVS * 5)

        self.assertFalse(np.isnan(obs).any(), msg="No NaN values in observations")
        self.assertFalse(np.isinf(obs).any(), msg="No Inf values in observations")
        self.assertFalse(np.isnan(state).any(), msg="No NaN values in state")
        self.assertFalse(np.isinf(state).any(), msg="No Inf values in state")

    def test_deterministic_behavior(self):
        """Test deterministic behavior with a fixed seed."""
        env1 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)
        env2 = MultiUAVMECEnv(num_users=NUM_USERS, num_uavs=NUM_UAVS, seed=42)

        obs1, state1 = env1.reset()
        obs2, state2 = env2.reset()

        # Simulate the same number of steps
        for _ in range(5):
            env1.step([0.0] * NUM_UAVS * 5)
            env2.step([0.0] * NUM_UAVS * 5)

        # Compare completed tasks
        tasks1 = env1.completed_tasks
        tasks2 = env2.completed_tasks
        self.assertEqual(len(tasks1), len(tasks2), msg="Number of completed tasks should match")
        for t1, t2 in zip(tasks1, tasks2):
            self.assertEqual(t1.task_id, t2.task_id, msg="Task IDs should match")
            self.assertEqual(t1.arrival_time, t2.arrival_time, msg="Arrival times should match")


if __name__ == "__main__":
    unittest.main()