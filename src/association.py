"""
User–UAV association (Phase 1: nearest feasible UAV).

Replace `associate_nearest` later with RL-based or chain-aware assignment
without changing the environment's task pipeline.
"""

import numpy as np


def associate_nearest(user_position, uavs):
    """
    Assign a user/task to the nearest UAV in 3D Euclidean distance.

    Parameters
    ----------
    user_position : array-like (3,)
    uavs : sequence of objects with `.position` and `.id`

    Returns
    -------
    uav_id : int
    distance : float
    """
    if not uavs:
        raise ValueError("associate_nearest requires at least one UAV")
    user_position = np.asarray(user_position, dtype=float)
    best_id = uavs[0].id
    best_d = float("inf")
    for uav in uavs:
        d = float(np.linalg.norm(user_position - np.asarray(uav.position, dtype=float)))
        if d < best_d:
            best_d = d
            best_id = uav.id
    return best_id, best_d
