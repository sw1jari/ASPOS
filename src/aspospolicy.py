import gymnasium as gym
import numpy as np
from typing import Optional, List, Tuple, Union
import torch

from koopman.env_model import KoopmanLinearModel
import osqp
import scipy.sparse as sp

class ASPOSCBFPolicy:
    """
    A safety shield using a Control Barrier Function (CBF) with a learned
    Koopman operator.

    This shield ensures safety by solving a small, efficient Quadratic Program (QP)
    at each timestep to find an action that satisfies the CBF condition, keeping
    the system within the safe set.
    """
    def __init__(
        self,
        env: Union[KoopmanLinearModel],
        state_space: gym.Space,
        ori_state_space: gym.Space,
        action_space: gym.Space,
        horizon: int,
        unsafe_polys: List[np.ndarray],
        safe_polys: List[np.ndarray],
        transform=lambda x: x,
        gamma=0.7,
        alpha=0.1
    ):
        """
        Args:
            koopman_model: The trained Koopman model with `transition` and `get_eps` methods.
            state_space: The latent (Koopman) state space.
            action_space: The environment's action space.
            cbf_gamma: A hyperparameter (0 < gamma < 1) that controls how quickly
                       the state is pushed away from the boundary.
            transform: A function to lift the state to the Koopman space.
        """

        self.env = env
        self.horizon = horizon
        self.state_space = state_space
        self.ori_state_space = ori_state_space
        self.action_space = action_space
        self.unsafe_polys = unsafe_polys
        self.safe_polys = safe_polys
        self.transform = transform

        self.s_dim = self.state_space.shape[0]
        self.u_dim = self.action_space.shape[0]
        # For caching results
        self.saved_state = None
        self.saved_action = None
        self.shielded = False
        self.gamma = gamma
        self._alpha = alpha

        # --- Placeholders for the pre-computed model and solver ---
        self.precomputed = {}
        self.is_model_updated = False

        # --- For caching results ---
        self.saved_state = None
        self.saved_action = None
        self.shielded = False

    def update_model(self):
        """
        Updates the shield with a new dynamics model. It pre-computes all
        state-independent components for EACH safe polyhedron provided.
        """
        H_max = self.horizon

        # Get a single, fixed dynamics model for pre-computation
        z_init = np.zeros(self.s_dim) # Use a zero state for linearization
        a_pi_init = np.zeros(self.u_dim)
        mat_dyn, adaptive_info = self.env.get_matrix_at_point(np.concatenate((z_init, a_pi_init)), self.s_dim)
        A, B, c = mat_dyn[:, :self.s_dim], mat_dyn[:, self.s_dim:-1], mat_dyn[:, -1]

        V, eps = adaptive_info
        V = np.asarray(V, float)
        eps_vec = np.asarray(eps, float).reshape(
            -1,
        )
        if eps_vec.size == 1:
            eps_vec = np.full(V.shape[1], eps_vec.item())

        # --- Pre-compute for EACH polyhedron ---
        self.solvers = []
        self.precomputed_per_poly = []

        # 1. Pre-compute powers and affine terms (same for all polys)
        A_pows = [np.eye(self.s_dim)]
        for _ in range(1, H_max + 1): A_pows.append(A_pows[-1] @ A)
        C_list = [np.zeros(self.s_dim)]
        for j in range(1, H_max + 1): C_list.append(C_list[-1] + A_pows[j - 1] @ c)

        for poly in self.safe_polys:

            # 2. Pre-compute face info for the current polyhedron
            #print(poly.shape)
            P_sel, b_sel = poly[:, :-1].astype(float), poly[:, -1].astype(float)
            faces = [(P_sel[i, :], float(b_sel[i])) for i in range(P_sel.shape[0])]

            def rel_degree(p: np.ndarray) -> Optional[int]:
                M = B.copy()
                for r in range(1, H_max + 1):
                    if np.linalg.norm(p @ M, ord=np.inf) > 1e-4:
                        return r
                    M = A @ M
                return None

            face_info = [(p, b, rel_degree(p)) for (p, b) in faces]
            r_vals = [r for (_, _, r) in face_info if r is not None]
            H_trap_all = max(r_vals) if r_vals else 1

            # 3. Pre-compute tightening terms
            tighten_pref = {
                tuple(p): np.cumsum(
                    [np.abs(p @ A_pows[ell]) @ eps_vec for ell in range(H_max)]
                )
                for p, _, _ in face_info
            }

            # 4. Build global constraint matrices for this polyhedron
            G_rows, M_h_rows, v_h_rows, row_ptrs = [], [], [], [[] for _ in range(H_max)]
            for j in range(1, H_max + 1):
                blocks = [A_pows[j - 1 - t] @ B for t in range(j)]
                phi_j = np.hstack(blocks + [np.zeros((self.s_dim, (H_max - j) * self.u_dim))])
                for p, b, r in face_info:
                    if r is None or j < r:
                        continue
                    G_rows.append(sp.csr_matrix(p @ phi_j))
                    M_h_rows.append(-p @ A_pows[j] + (self.gamma ** j) * p)
                    tighten = tighten_pref[tuple(p)][j - 1]
                    pomdp_factor = self._alpha * np.linalg.norm(p, ord=1)
                    v_h_rows.append(-p @ C_list[j] - b - tighten + (self.gamma ** j) * (b - pomdp_factor) - pomdp_factor)
                    row_ptrs[j - 1].append(len(G_rows) - 1)

            G_all = (
                sp.vstack(G_rows, format="csc")
                if G_rows
                else sp.csc_matrix((0, H_max * self.u_dim))
            )
            M_h = np.vstack(M_h_rows) if M_h_rows else np.zeros((0, self.s_dim))
            v_h = np.array(v_h_rows)

            # 5. Setup a dedicated OSQP solver for this polyhedron
            P_blocks = [sp.eye(self.u_dim, format="csc")] + [
                1e-4 * sp.eye(self.u_dim, format="csc")
            ] * (H_max - 1)
            Pmat = sp.block_diag(P_blocks, format="csc")
            lb_actions = np.tile(self.action_space.low, H_max)
            ub_actions = np.tile(self.action_space.high, H_max)
            A_qp = sp.vstack([G_all, sp.eye(H_max * self.u_dim, format="csc")], format="csc")
 
            solver = osqp.OSQP()
            solver.setup(P=Pmat, q=np.zeros(H_max * self.u_dim), A=A_qp,
                          l=np.hstack([-np.inf * np.ones(G_all.shape[0]), lb_actions]),
                          u=np.hstack([np.zeros(G_all.shape[0]), ub_actions]),
                          verbose=False, polish=False)

            self.solvers.append(solver)
            self.precomputed_per_poly.append({
                "M_h": M_h, "v_h": v_h, "row_ptrs": row_ptrs,
                "H_trap_all": H_trap_all, "ub_actions": ub_actions
            })

        self.is_model_updated = True
        #print(f"[RAMPS] Model updated. Pre-computed constraints for {len(self.safe_polys)} polyhedra.")

    def solve(
        self,
        state: np.ndarray,
        action: Optional[np.ndarray] = None,
        debug: bool = False,
    ) -> Tuple[np.ndarray, bool]:
        """
        Solves for a safe action by first selecting the most appropriate safe
        polyhedron and then using its dedicated pre-computed solver.
        """
        if not self.is_model_updated:
            #print("Model not initialized. Performing first-time update.")
            self.update_model()

        z = self.transform(state.reshape(1, -1)).reshape(-1,)
        a_pi = np.zeros(self.u_dim, dtype=float) if action is None else np.asarray(action, float)
        # --- 1. Choose the Active Polyhedron based on current state z ---
        inside_candidates, violated = [], []
        for idx, poly in enumerate(self.safe_polys):
            P, b = poly[:, :-1].astype(float), poly[:, -1].astype(float)
            g = P @ z + b
            
            #print(f"Poly {idx}, min margin {-g.max()} at face, {g.argmax()}")
            worst = float(np.max(g))
            if worst <= 1e-9: # Use a small tolerance
                inside_candidates.append((idx, worst))
            else:
                violated.append((idx, worst))

        if inside_candidates:
            # Pick the poly we are IN with the largest interior slack
            chosen_idx = min(inside_candidates, key=lambda t: t[1])[0]
            mode = "inside"
        elif violated:
            # Fallback: pick the most violated poly
            chosen_idx = min(violated, key=lambda t: t[1])[0]
        else:
            # Should not happen if safe_polys is not empty
            #print("[RAMPS] WARN: No safe polyhedra found for the current state.")
            return self.backup(state)

        #if debug: print(f"[RAMPS] selection mode={mode}, chosen poly idx={chosen_idx}")

        # --- 2. Select the correct pre-computed solver and data ---
        solver = self.solvers[chosen_idx]
        data = self.precomputed_per_poly[chosen_idx]

        # --- 3. State-Dependent Calculations ---
        h_all = data['M_h'] @ z + data['v_h']

        # --- 4. Binary Search for the Largest Feasible Horizon ---
        lo, hi = data['H_trap_all'], self.horizon
        bestH, best_u0, best_dev = 0, None, None
        q_new = np.hstack([-a_pi, np.zeros((self.horizon - 1) * self.u_dim)])
        u_base = np.hstack([np.zeros_like(h_all), data['ub_actions']])

        while lo <= hi:
            mid = (lo + hi) // 2
            active_rows = [idx for j in range(mid) for idx in data['row_ptrs'][j]]
            mask = np.full_like(h_all, np.inf)
            if active_rows:
                mask[active_rows] = h_all[active_rows]
            u_new = u_base.copy()
            u_new[:len(mask)] = mask

            solver.update(q=q_new, u=u_new)
            res = solver.solve()

            if res.info.status == "solved":
                bestH, best_u0, best_dev = (
                    mid,
                    res.x[: self.u_dim],
                    float(np.linalg.norm(res.x[: self.u_dim] - a_pi)),
                )
                lo = mid + 1
            else:
                hi = mid - 1

        # --- 5. Return Result or Fallback ---
        if bestH > 0:
            shielded = best_dev > 1e-8
            #if debug: print(f"[RAMPS] Solved. Largest feasible H={bestH}, ||u0-a_pi||={best_dev:.3e}")
            self.saved_state, self.saved_action, self.shielded = state, best_u0, shielded
            return best_u0, shielded

        #if debug:
            #print("[RAMPS] No feasible H found. Using backup policy.")
        u0 = self.backup(state)
        self.saved_state, self.saved_action, self.shielded = state, u0, False
        return u0, False

    def backup(self, state: np.ndarray) -> np.ndarray:
        """
        A robust backup policy that actively steers the system towards safety.
        It finds the most critical safety constraint and chooses an action that
        maximally increases the corresponding barrier function's value.
        """
        z = self.transform(state.reshape(1, -1)).reshape(-1,)
        s_dim = self.state_space.shape[0]
        u_dim = self.action_space.shape[0]

        # 1. Find the most critical safety constraint (the one we are closest to violating)
        min_h_val = np.inf
        most_critical_grad = None
        for poly in self.safe_polys:
            P_poly, b_poly = poly[:, :-1], poly[:, -1]
            for i in range(P_poly.shape[0]):
                p_i, b_i = P_poly[i, :], b_poly[i]
                # h_i(z) = -(p_i^T * z + b_i)
                h_i_z = -(p_i @ z + b_i)
                if h_i_z < min_h_val:
                    min_h_val = h_i_z
                    # The gradient ∇h_i(z) = -p_i points "inward" toward safety
                    most_critical_grad = -p_i

        if most_critical_grad is None:
            # This can happen if the state is somehow outside all defined safe polytopes.
            # Returning a zero action is a reasonable passive fallback.
            return np.zeros(u_dim)

        # 2. Get the B matrix from the Koopman model (linearized around a zero action)
        mat_dyn, _ = self.env.get_matrix_at_point(np.concatenate((z, np.zeros(u_dim))), s_dim)
        B = mat_dyn[:, s_dim:-1]

        # 3. Formulate and solve a QP to find the best recovery action
        # Objective: Find action 'u' that maximizes the rate of safety increase,
        # which is equivalent to maximizing ∇h^T * (Bu).
        # min - (∇h^T * B) * u
        P_backup = sp.csc_matrix((u_dim, u_dim)) # No quadratic term
        q_backup = -(most_critical_grad.T @ B)

        # Constraints are just the action bounds
        A_backup = sp.csc_matrix(sp.eye(u_dim))
        l_backup = self.action_space.low
        u_backup = self.action_space.high

        solver = osqp.OSQP()
        solver.setup(P=P_backup, q=q_backup, A=A_backup, l=l_backup, u=u_backup, verbose=False)
        res = solver.solve()

        if res.info.status == 'solved':
            return res.x
        else:
            # If the recovery QP fails (should be rare), return a passive action
            #print("WARN: Backup recovery QP failed. Returning zero action.")
            return np.zeros(u_dim)

    def __call__(self, state: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Main entry point for the shield.
        """
        if self.saved_state is not None and np.allclose(state, self.saved_state):
            return self.saved_action, self.shielded
        return self.solve(state)

    def unsafe(self, state: np.ndarray, action: np.ndarray) -> bool:
        """
        Checks if a proposed action is unsafe by seeing if the shield would modify it.
        """
        safe_action, shielded = self.solve(state, action=action)
        return np.linalg.norm(safe_action - action) > 1e-8
