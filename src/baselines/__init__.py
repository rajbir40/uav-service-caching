# src/baselines/__init__.py
# ----------------------------------------------------------------
# Baseline agents for comparison against DMJO (Ours).
# All agents expose the same interface as MAPPOAgent:
#   get_actions(obs_all, state, num_active)  → actions, log_probs, values
#   get_values(state)                        → values
#   update(buffer)                           → metrics dict
#   save(path) / load(path)
#
# Import via the factory:
#   from src.agent_factory import make_agent
# ----------------------------------------------------------------
from src.baselines.mappo_baseline   import MAPPOBaseline
from src.baselines.maddpg_agent     import MADDPGAgent
from src.baselines.matd3_agent      import MATD3Agent
from src.baselines.greedy_agent     import GreedyOffloadingAgent
from src.baselines.local_agent      import LocalExecutionAgent

__all__ = [
    "MAPPOBaseline",
    "MADDPGAgent",
    "MATD3Agent",
    "GreedyOffloadingAgent",
    "LocalExecutionAgent",
]
