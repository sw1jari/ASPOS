import numpy as np

SQRT_3 = np.sqrt(3)
class NoisyObservationWrapper:
    """Wraps a gym environment to add uniform noise to the observations.
    Returns o = s + v with v ~ U(-alpha, alpha) per dim, true state s goes in info dict."""
    def __init__(self, env, alpha, seed, n_type="uniform"):
        self.env = env
        self.alpha = alpha
        self.rng = np.random.default_rng(seed)
        self.noise_type = n_type


    def observe(self, state, info):
        info['true_state'] = state
        if self.noise_type == 'uniform':
            noise = self.rng.uniform(-self.alpha, self.alpha, size=state.shape)
        elif self.noise_type == 'gaussian':
            noise = self.rng.normal(0, self.alpha/SQRT_3, size=state.shape)
        return state + noise, info

    def reset(self, seed=None):
        state, info = self.env.reset(seed=seed)
        return self.observe(state, info)

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)
        obs, info = self.observe(state, info)
        return obs, reward, done, trunc, info

    def __getattr__(self, name):
        """Pass through attributes to the wrapped environment."""
        return getattr(self.env, name)