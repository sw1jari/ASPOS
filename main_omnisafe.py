# This runs our envs on OmniSafe, which can run CMDP baselines from the RAMPS paper like CUP and
# P3O. Also PPO-Lagrangian (here called PPOLag), and other things.
#
# SETUP
# Version 0.5.0 is the latest release, it needs python<=3.11 and numpy<2.
# Use uv (what i did):
# uv venv -p 3.10 .venv-omnisafe && source .venv-omnisafe/bin/activate
# uv pip install omnisafe==0.5.0 "numpy<2"

# RUNNING THIS
# Then you can run this with --algo PPOLag, CUP or P3O, for example:
# python main_omnisafe.py --algo CUP --env_name pendulum --noise 0.1 --seed 1
# There's also omnisafe_exps.sh, which allows you to run experiments in parallel

import argparse
import csv
import datetime
import os

import omnisafe
import torch
from omnisafe.envs.core import CMDP, env_register

from benchmarks import envs
from benchmarks.noisy_obs import NoisyObservationWrapper

ENVS = ['pendulum', 'cheetah', 'hopper', 'ant']


# This class makes our env with observation noise an OmniSafe env that its algorithms can train on
# @env_register adds it to OmniSafe's envs, so omnisafe.Agent can build it from Noisy-pendulum etc
# For that it subclasses CMDP, lists its ids in _support_envs and defines step, reset, set_seed,
# render and close, see https://omnisafe.readthedocs.io/en/latest/envs/core.html
@env_register
class NoisyOmniSafeEnv(CMDP):
    _support_envs = [f'Noisy-{name}' for name in ENVS]
    # Our envs end their own episodes at the time limit, OmniSafe only has to reset them
    need_time_limit_wrapper = False
    need_auto_reset_wrapper = True
    # Noise, seed and log path, set by main() because only PPOLag's config can pass them to __init__
    settings = {}

    # OmniSafe also passes num_envs and device, which we do not use
    def __init__(self, env_id, **kwargs):
        # Our env behind the noise wrapper, Noisy-pendulum becomes pendulum
        env = envs.get_env_from_name(env_id.split('-', 1)[1])
        self.env = NoisyObservationWrapper(env, self.settings['noise'], self.settings['seed'])
        # OmniSafe reads these through num_envs, observation_space and action_space
        self._num_envs = 1
        self._observation_space = self.env.observation_space
        self._action_space = self.env.action_space
        self.seed = self.settings['seed']
        self.writer = None
        self.total_steps = 0
        self.start_episode()

    # Resets the counters of the current episode
    def start_episode(self):
        self.episode_steps = 0
        self.episode_reward = 0.0
        self.unsafe = False

    # This logs one row per episode to episodes.csv, which we need for the unsafe and reward totals
    def log_episode(self, cut):
        # Open the file and write the header at the first episode
        if self.writer is None:
            self.file = open(self.settings['csv_path'], 'w', newline='')
            self.writer = csv.writer(self.file)
            self.writer.writerow(['total_steps', 'episode_steps', 'reward', 'unsafe', 'cut'])
        row = [self.total_steps, self.episode_steps, self.episode_reward, int(self.unsafe), int(cut)]
        self.writer.writerow(row)
        self.file.flush()

    # Steps our env with OmniSafe's action and gives cost 1 when the true state is unsafe
    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action.numpy())
        self.total_steps += 1
        self.episode_steps += 1
        # Unsafe true state gives cost 1, -100 reward and ends the episode, like the shielded runs
        # The -100 stays in episodes.csv, combined_main.py logs its reward without it
        cost = 0.0
        if self.env.unsafe(info['true_state'], False):
            cost = 1.0
            reward -= 100
            done = True
            self.unsafe = True
        # Log the episode once it ends
        self.episode_reward += reward
        if done or trunc:
            self.log_episode(cut=False)
            self.start_episode()
        # OmniSafe works on tensors, its networks need float32 states and our env gives float64
        state = torch.as_tensor(state, dtype=torch.float32)
        return state, torch.as_tensor(reward), torch.as_tensor(cost), torch.as_tensor(done), torch.as_tensor(trunc), {}

    # OmniSafe's wrappers pass seed and options
    def reset(self, seed=None, options=None):
        # A reset mid episode is OmniSafe starting a new epoch, that episode is logged as cut
        if self.episode_steps > 0:
            self.log_episode(cut=True)
        self.start_episode()
        # Seed only the first reset, like combined_main.py
        state, _ = self.env.reset(seed=self.seed)
        self.seed = None
        return torch.as_tensor(state, dtype=torch.float32), {}

    # OmniSafe calls this with the run's seed before the first reset
    def set_seed(self, seed):
        self.seed = seed

    # CMDP requires render, without it the class cannot be made, but we never render
    def render(self):
        pass

    # CMDP leaves close to us, OmniSafe calls it after training, so we close episodes.csv there
    def close(self):
        if self.writer is not None:
            self.file.close()


def main(args):
    # Run folder with time, algorithm and settings, like our other runs
    time = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    name = f'{time}_{args.algo}_{args.env_name}_S{args.seed}_N{args.noise}_C{args.cost_limit}'
    out = os.path.join('runs', name)
    os.makedirs(out, exist_ok=True)

    # Set settings for NoisyOmniSafeEnv, which is built by omnisafe.Agent from the env name
    csv_path = os.path.join(out, 'episodes.csv')
    NoisyOmniSafeEnv.settings = {'noise': args.noise, 'seed': args.seed, 'csv_path': csv_path}

    # OmniSafe defaults except run length, one thread per run, rollout length and log folder
    cfgs = {
        'seed': args.seed,
        'train_cfgs': {'total_steps': args.num_steps, 'torch_threads': 1},
        'algo_cfgs': {'steps_per_epoch': args.steps_per_epoch},
        'logger_cfgs': {'log_dir': out},
    }
    # P3O keeps its cost limit with the algorithm settings, PPOLag and CUP with the Lagrangian ones
    if args.algo == 'P3O':
        cfgs['algo_cfgs']['cost_limit'] = args.cost_limit
    else:
        cfgs['lagrange_cfgs'] = {'cost_limit': args.cost_limit}

    # Here we pass the env name to omnisafe.Agent, which builds NoisyOmniSafeEnv from it
    agent = omnisafe.Agent(args.algo, f'Noisy-{args.env_name}', custom_cfgs=cfgs)

    # Train the agent
    agent.learn()

    # Totals of the training env in the format of our other runs, without the cut episodes
    rows = list(csv.DictReader(open(csv_path)))
    done = [row for row in rows if row['cut'] == '0']
    unsafe = sum(int(row['unsafe']) for row in done)
    print(f'Total unsafe real: {unsafe} / {len(done)}')
    print(f'Cut by an epoch end: {len(rows) - len(done)}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', default='PPOLag', choices=['PPOLag', 'CUP', 'P3O'])
    parser.add_argument('--env_name', default='pendulum', choices=ENVS)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--num_steps', type=int, default=200000)
    # Steps per update, 2048 like the batch size of our PPO runs
    parser.add_argument('--steps_per_epoch', type=int, default=2048)
    # The algorithms keep the average cost per episode below this limit. An unsafe step costs 1
    # and ends the episode, so an episode costs 0 or 1 and the average is the share of unsafe
    # episodes. 0.01 allows 1% unsafe episodes, the same 1% risk as the 99th percentile error
    # bound in RAMPS.
    parser.add_argument('--cost_limit', type=float, default=0.01)
    main(parser.parse_args())
