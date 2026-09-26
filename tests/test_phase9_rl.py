#!/usr/bin/env python3
"""
Phase 9 RL Tests
==================

Tests for MAPPO and MADDPG integration in the UAV-MEC system.
"""

import unittest
import numpy as np
import torch
from src.mappo_agent import MAPPOAgent
from src.madrl_utils import RolloutBuffer, compute_gae, compute_returns, compute_advantages
from config import (
    NUM_USERS, NUM_UAVS, OBS_DIM, STATE_DIM, ACTION_DIM,
    MAPPO_LR_ACTOR, MAPPO_LR_CRITIC, MAPPO_CLIP_EPS,
    MAPPO_ENTROPY_COEF, MAPPO_VALUE_COEF, MAPPO_MAX_GRAD_NORM,
    MAPPO_PPO_EPOCHS, MAPPO_NUM_MINI_BATCHES,
    MAPPO_HIDDEN_DIM, MAPPO_CRITIC_HIDDEN_DIM,
    USE_DIFFUSION
)


class TestMAPPOAgent(unittest.TestCase):
    """Test MAPPO agent."""

    def setUp(self):
        self.device = 'cpu'
        self.agent = MAPPOAgent(
            obs_dim=OBS_DIM,
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            n_agents=NUM_UAVS,
            device=self.device
        )

    def test_agent_initialization(self):
        """Test that the agent initializes correctly."""
        self.assertEqual(self.agent.obs_dim, OBS_DIM)
        self.assertEqual(self.agent.state_dim, STATE_DIM)
        self.assertEqual(self.agent.action_dim, ACTION_DIM)
        self.assertEqual(self.agent.n_agents, NUM_UAVS)
        self.assertEqual(self.agent.device, self.device)

    def test_get_actions(self):
        """Test that the agent can generate actions."""
        obs_all = np.random.randn(NUM_UAVS, OBS_DIM)
        state = np.random.randn(STATE_DIM)

        actions, log_probs, values = self.agent.get_actions(obs_all, state)
        self.assertEqual(actions.shape, (NUM_UAVS, ACTION_DIM))
        self.assertEqual(log_probs.shape, (NUM_UAVS,))
        self.assertEqual(values.shape, (NUM_UAVS,))

    def test_get_values(self):
        """Test that the agent can compute values."""
        state = np.random.randn(STATE_DIM)
        values = self.agent.get_values(state)
        self.assertEqual(values.shape, (NUM_UAVS,))


class TestRolloutBuffer(unittest.TestCase):
    """Test RolloutBuffer."""

    def setUp(self):
        self.max_size = 1000
        self.obs_dim = OBS_DIM
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.n_agents = NUM_UAVS
        self.buffer = RolloutBuffer(
            max_size=self.max_size,
            obs_dim=self.obs_dim,
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            n_agents=self.n_agents
        )

    def test_buffer_initialization(self):
        """Test that the buffer initializes correctly."""
        self.assertEqual(len(self.buffer.obs_buffer), 0)
        self.assertEqual(len(self.buffer.action_buffer), 0)
        self.assertEqual(len(self.buffer.reward_buffer), 0)
        self.assertEqual(len(self.buffer.next_obs_buffer), 0)
        self.assertEqual(len(self.buffer.done_buffer), 0)
        self.assertEqual(len(self.buffer.state_buffer), 0)
        self.assertEqual(len(self.buffer.agent_id_buffer), 0)
        self.assertEqual(len(self.buffer.old_log_prob_buffer), 0)
        self.assertEqual(len(self.buffer.old_value_buffer), 0)

    def test_store_and_retrieve(self):
        """Test storing and retrieving transitions."""
        obs = np.random.randn(self.obs_dim)
        action = np.random.randn(self.action_dim)
        reward = np.random.randn()
        next_obs = np.random.randn(self.obs_dim)
        done = np.random.randint(0, 2)
        state = np.random.randn(self.state_dim)
        agent_id = np.random.randint(0, self.n_agents)
        old_log_prob = np.random.randn()
        old_value = np.random.randn()

        self.buffer.store(
            obs, action, reward, next_obs, done, state, agent_id, old_log_prob, old_value
        )

        self.assertEqual(len(self.buffer.obs_buffer), 1)
        self.assertEqual(len(self.buffer.action_buffer), 1)
        self.assertEqual(len(self.buffer.reward_buffer), 1)
        self.assertEqual(len(self.buffer.next_obs_buffer), 1)
        self.assertEqual(len(self.buffer.done_buffer), 1)
        self.assertEqual(len(self.buffer.state_buffer), 1)
        self.assertEqual(len(self.buffer.agent_id_buffer), 1)
        self.assertEqual(len(self.buffer.old_log_prob_buffer), 1)
        self.assertEqual(len(self.buffer.old_value_buffer), 1)

    def test_get_batches(self):
        """Test getting mini-batches from the buffer."""
        # Fill the buffer with random data
        for _ in range(10):
            obs = np.random.randn(self.obs_dim)
            action = np.random.randn(self.action_dim)
            reward = np.random.randn()
            next_obs = np.random.randn(self.obs_dim)
            done = np.random.randint(0, 2)
            state = np.random.randn(self.state_dim)
            agent_id = np.random.randint(0, self.n_agents)
            old_log_prob = np.random.randn()
            old_value = np.random.randn()

            self.buffer.store(
                obs, action, reward, next_obs, done, state, agent_id, old_log_prob, old_value
            )

        # Get mini-batches
        batches = list(self.buffer.get_batches(num_mini_batches=2))
        self.assertEqual(len(batches), 2)

        for batch in batches:
            self.assertIn('obs', batch)
            self.assertIn('actions', batch)
            self.assertIn('rewards', batch)
            self.assertIn('next_obs', batch)
            self.assertIn('dones', batch)
            self.assertIn('states', batch)
            self.assertIn('agent_ids', batch)
            self.assertIn('old_log_probs', batch)
            self.assertIn('old_values', batch)
            self.assertIn('advantages', batch)
            self.assertIn('returns', batch)


class TestGAE(unittest.TestCase):
    """Test Generalized Advantage Estimation (GAE)."""

    def test_compute_gae(self):
        """Test GAE computation."""
        rewards = np.random.randn(10)
        dones = np.zeros(10)
        values = np.random.randn(10)

        advantages, returns = compute_gae(rewards, dones, values)
        self.assertEqual(advantages.shape, (10,))
        self.assertEqual(returns.shape, (10,))

    def test_compute_returns(self):
        """Test returns computation."""
        rewards = np.random.randn(10)
        dones = np.zeros(10)

        returns = compute_returns(rewards, dones)
        self.assertEqual(returns.shape, (10,))

    def test_compute_advantages(self):
        """Test advantages computation."""
        rewards = np.random.randn(10)
        dones = np.zeros(10)
        values = np.random.randn(10)

        advantages = compute_advantages(rewards, dones, values)
        self.assertEqual(advantages.shape, (10,))


if __name__ == "__main__":
    unittest.main()