
import gymnasium as gym
from pytorch_soft_actor_critic.sac import SAC
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
from ppo import PPO
import numpy as np

class SACPolicy:
    def __init__(
        self, gym_env: gym.Env, replay_size: int, seed: int, batch_size: int, sac_args
    ):
        self.agent = SAC(
            gym_env.observation_space.shape[0], gym_env.action_space, sac_args
        )
        self.memory = ReplayMemory(
            replay_size, gym_env.observation_space, gym_env.action_space.shape[0], seed
        )
        self.updates = 0
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state, evaluate = evaluate)

    def add(self, state, action, reward, next_state, done, cost):
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self):
        ret = self.agent.update_parameters(self.memory, self.batch_size, self.updates)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)


class PPOPolicy:
    def __init__(
        self, gym_env: gym.Env, replay_size: int, seed: int, batch_size: int, args
    ):
        self.agent = PPO(gym_env.observation_space.shape[0], gym_env.action_space, args)
        self.memory = ReplayMemory(
            replay_size, gym_env.observation_space, gym_env.action_space.shape[0], seed
        )
        self.updates = 0
        self.minibatch_size = args.mini_batch_size
        self.batch_size = batch_size

    def __call__(self, state: np.ndarray, evaluate: bool = False):
        return self.agent.select_action(state)[0]

    def add(self, state, action, reward, next_state, done, cost):
        self.memory.push(state, action, reward, next_state, done, cost)

    def train(self):
        ret = self.agent.update_parameters(self.memory, batch_size=self.minibatch_size, epochs = 40)
        self.updates += 1
        return ret

    def report(self):
        return 0, 0

    def load_checkpoint(self, path):
        self.agent.load_checkpoint(path)

