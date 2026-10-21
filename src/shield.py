
import numpy as np
from typing import Tuple
import time

class Shield:
    """
    Construct a shield from a neural policy and a safety layer.
    """

    def __init__(
        self,
        shield_policy,
        unsafe_policy=None,
        means: np.ndarray = None,
        stds: np.ndarray = None,
    ):
        self.shield = shield_policy
        self.agent = unsafe_policy
        self.shield_times = 0
        self.backup_times = 0
        self.agent_times = 0
        self.total_time = 0.0
        self.means = means
        self.stds = stds

    def __call__(self, state: np.ndarray, action: np.ndarray = None, **kwargs) -> np.ndarray:
        start = time.time()
        if action is not None:
            proposed_action = action
        else:
            proposed_action = self.agent(state, **kwargs)

        if self.means is not None:
            state = (state - self.means) / self.stds

        if self.shield.unsafe(state, proposed_action):
            act, shielded = self.shield(state)
            self.shield_times += 1 if shielded else 0
            self.backup_times += 1 if not shielded else 0
            shielded = "SHIELD" if shielded else "BACKUP"
        else:
            act = proposed_action
            shielded = "NEURAL"
            self.agent_times += 1
        end = time.time()
        self.total_time += end - start

        # print(f"Shield: {shielded}, Action: {act}, Time: {end - start:.4f}s")
        return act, shielded, np.linalg.norm(act - proposed_action), proposed_action

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0
