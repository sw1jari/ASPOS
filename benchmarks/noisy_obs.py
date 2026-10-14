import numpy as np


class NoisyObservation:
    """Wraps an env so it returns o = s + v, with the true state s in info['true_state']."""
    def __init__(self, env, dist, scale, seed):
        self.env = env
        self.dist = dist
        self.scale = scale
        self.rng = np.random.default_rng(seed)

    def _observe(self, state, info):
        """Adds noise to the true state and returns it."""
        # Add noise to the true state
        if self.dist == 'uniform':
            noise = self.rng.uniform(-self.scale, self.scale, size=state.shape)
        else:
            noise = self.rng.normal(0.0, self.scale, size=state.shape)
        return state + noise.astype(state.dtype), {**info, 'true_state': state}

    def reset(self, **kwargs):
        return self._observe(*self.env.reset(**kwargs))

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        obs, info = self._observe(state, info)
        return obs, reward, done, trunc, info

    def __getattr__(self, name):
        """Forward attributes to the underlying env, except for 'env' itself to avoid recursion."""
        if name == 'env':
            raise AttributeError(name)
        return getattr(self.env, name)
