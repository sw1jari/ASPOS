
import numpy as np
import scipy.linalg
from typing import Tuple
import time

class KalmanEstimate:
    # mat is the linear model [A | B | c] for x_next = A x + B u + c
    # residuals are the one-step model errors on the normalised observations
    # R is the covariance of the observation noise.
    def __init__(self, mat, residuals, R):
        n = mat.shape[0]
        self.A, self.B, self.c = mat[:, :n], mat[:, n:-1], mat[:, -1]
        # Eq. (9): Q is the covariance of the model noise w
        # Ours: Cov(residuals) = Q + A R A^T + R, solve for Q and floor its eigenvalues so it stays a covariance
        vals, vecs = np.linalg.eigh(np.cov(residuals.T) - self.A @ R @ self.A.T - R)
        Q = vecs @ np.diag(np.maximum(vals, 1e-6)) @ vecs.T
        # Eq. (14), (16): Sigma_dot = A Sigma + Sigma A^T + Q, at an observation Sigma+ = (I - K H) Sigma-
        # Ours: H = I and fixed A, Q, R, so the prior Sigma- settles at P = A (I - K) P A^T + Q
        # We use the discrete algebraic Riccati equation to solve for P, then K = P (P + R)^-1
        P = scipy.linalg.solve_discrete_are(self.A.T, np.eye(n), Q, R)
        # Eq. (16): K = Sigma- H^T (H Sigma- H^T + R)^-1
        # Ours: H = I and Sigma- = P
        self.K = P @ np.linalg.inv(P + R)
        self.P, self.Q, self.R = P, Q, R
        # Eq. (5): belief b = (mu, Sigma)
        # Ours: Sigma is fixed so only mu is kept, None before an episode's first observation
        self.mu = None

    def __call__(self, obs, prev_action):
        # The first observation of an episode is taken as is
        if self.mu is not None:
            # Eq. (13): mu_dot = f(mu) + g(mu) u
            # Ours: one step mu- = A mu + B u + c
            pred = self.A @ self.mu + self.B @ prev_action + self.c
            # Eq. (16): mu+ = mu- + K (z_k - l(mu-))
            # Ours: l(mu) = mu
            obs = pred + self.K @ (obs - pred)
        self.mu = obs
        return obs


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
        estimate=None,
    ):
        self.shield = shield_policy
        # Optional state estimate the shield acts on instead of the raw observation
        self.estimate = estimate
        self.prev_action = None
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
        if self.estimate is not None:
            state = self.estimate(state, self.prev_action)

        if self.shield.unsafe(state, proposed_action):
            act, shielded = self.shield(state)
            self.shield_times += 1 if shielded else 0
            self.backup_times += 1 if not shielded else 0
            shielded = "SHIELD" if shielded else "BACKUP"
        else:
            act = proposed_action
            shielded = "NEURAL"
            self.agent_times += 1
        # The next prediction A mu + B u + c needs the action that was applied
        self.prev_action = act
        end = time.time()
        self.total_time += end - start

        # print(f"Shield: {shielded}, Action: {act}, Time: {end - start:.4f}s")
        return act, shielded, np.linalg.norm(act - proposed_action), proposed_action

    def report(self) -> Tuple[int, int]:
        return self.shield_times, self.agent_times, self.backup_times, self.total_time

    def reset_count(self):
        # Runs at the end of every episode, so the estimate restarts from the next observation
        if self.estimate is not None:
            self.estimate.mu = None
        self.shield_times = 0
        self.agent_times = 0
        self.backup_times = 0
        self.total_time = 0
