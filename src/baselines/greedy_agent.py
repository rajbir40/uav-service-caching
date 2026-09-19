"""
Greedy Offloading Baseline  (Table I — Row 5)
===============================================
Nearest-UAV association without caching awareness.

Policy (rule-based, no learning):
  Each IoT device selects the geographically closest UAV and always
  offloads its task there, regardless of:
    • UAV load / queue depth
    • Cached program availability (cache-unaware)
    • Channel quality or interference
    • NOMA pairing opportunities

  UAV trajectories are fixed circular orbits (no learned movement).

This gives a strong greedy upper bound on simple distance-based
offloading and a lower bound on cache-aware methods.

Interface (identical to MAPPOAgent):
    get_actions(obs_all, state, num_active) → actions, log_probs(=0), values(=0)
    get_values(state)                       → zeros
    update(buffer)                          → empty metrics  (no learning)
    save(path) / load(path)                 → no-ops

Action encoding:
  The env's agent_action_dim vector is set to [-1, …, -1] (all zeros
  after tanh), signalling the env wrapper to apply the greedy override.
  The greedy logic itself lives in env.py's _greedy_offload_override()
  method, which this agent activates by setting the GREEDY_MODE flag.

  If env.py does not implement _greedy_offload_override(), actions are
  passed through as-is and the env uses its standard offloading logic
  driven by the all-zeros action vector (minimum-effort signal).
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


class GreedyOffloadingAgent:
    """
    Greedy Offloading baseline (Row 5).

    Rule-based: always emits the action vector that signals "offload to
    nearest UAV" to the environment.  No neural network, no training.
    """

    METHOD = "GreedyOffloading"

    def __init__(self, obs_dim, state_dim, action_dim, n_agents,
                 device="cpu", share_actor=False):
        self.obs_dim    = obs_dim
        self.state_dim  = state_dim
        self.action_dim = action_dim
        self.n_agents   = n_agents
        self.device     = device
        self.total_updates = 0

        # Greedy action: zeros map to tanh(0)=0, interpreted by env as
        # "minimum effort / nearest association". Env must handle the rest.
        self._greedy_action = np.zeros(action_dim, dtype=np.float32)

        # Expose a flag so the env can detect this agent type
        self.GREEDY_MODE = True

    # ----------------------------------------------------------
    # Action selection  (rule-based, O(1))
    # ----------------------------------------------------------
    def get_actions(self, obs_all, state, num_active=None):
        """
        Return identical all-zero actions for all agents.
        The environment's greedy override will resolve the actual
        nearest-UAV assignment per IoT device each time step.
        """
        n_active  = num_active or self.n_agents
        actions   = np.zeros((self.n_agents, self.action_dim), dtype=np.float32)
        log_probs = np.zeros(self.n_agents, dtype=np.float32)
        values    = np.zeros(self.n_agents, dtype=np.float32)
        # Only fill active agents — padded slots remain zero
        for i in range(n_active):
            actions[i] = self._greedy_action
        return actions, log_probs, values

    def get_values(self, state):
        return np.zeros(self.n_agents, dtype=np.float32)

    # ----------------------------------------------------------
    # No learning — update is a no-op
    # ----------------------------------------------------------
    def update(self, buffer=None) -> dict:
        """Rule-based agent: no gradient updates."""
        return dict(actor_loss=0.0, critic_loss=0.0,
                    entropy=0.0, approx_kl=0.0)

    # ----------------------------------------------------------
    # Checkpoint (nothing to save)
    # ----------------------------------------------------------
    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        import json
        meta = {'method': self.METHOD, 'total_updates': self.total_updates}
        with open(path.replace('.pt', '.json'), 'w') as f:
            json.dump(meta, f, indent=2)

    def load(self, path: str):
        pass   # no parameters to restore
