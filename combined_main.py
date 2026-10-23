import os
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
import argparse
import random
import datetime
import itertools
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from src import Shield, PPOPolicy, SACPolicy, ProjectionPolicy, CBFPolicy, ASPOSCBFPolicy
from koopman.env_model import get_environment_model
from constraints import safety
from benchmarks import envs
from benchmarks.noisy_obs import NoisyObservationWrapper
from pytorch_soft_actor_critic.replay_memory import ReplayMemory
import gymnasium as gym
import matplotlib.pyplot as plt
import imageio
import traceback
torch.set_num_threads(1)
import time

# Setup environment

def run_experiment(args):
    env = envs.get_env_from_name(args.env_name)
    env = NoisyObservationWrapper(env, args.noise, args.seed, args.noise_type)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Tensorboard
    if not os.path.exists("runs"):
        os.makedirs("runs")

    # Save folder naming
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_params = f"H{args.horizon}_S{args.seed}_N{args.noise}"
    safe_suffix = '_safe' if not args.no_safety else ''
    name = f"{timestamp}_{'ASPOS' if args.aspos else 'RAMPS'}_{'PPO' if args.ppo else 'SAC'}_{args.env_name}_{run_params}{safe_suffix}"
    writer = SummaryWriter(f'runs/{name}')

    hyperparams = vars(args)
    if args.verbosity > 2: print(hyperparams)

    file = open(f'runs/{name}/log.txt', "w+")
    os.makedirs(f'runs/{name}/videos')

    # Agent setup

    # Replay memories
    real_data = ReplayMemory(args.replay_size, env.observation_space, env.action_space.shape[0], args.seed)

    iterator_loop = itertools.count(1)

    real_unsafe_episodes = 0
    total_real_episodes = 0
    total_numsteps = 0
    if args.ppo:
        agent = PPOPolicy(env, args.batch_size, args.seed, args.batch_size, args)
    else:
        agent = SACPolicy(env, args.replay_size, args.seed, args.batch_size, args)
    safe_agent = None

    # Training loop

    env_model = None
    unsafe_test_episodes = 0
    total_test_episodes = 0
    unsafe_sim_episodes = 0
    total_sim_episodes = 0
    train_steps = 1

    avg_deviation = []
    while True:
        i_episode = next(iterator_loop)
        episode_reward = 0
        episode_steps = 0
        done = False
        trunc = False
        # Seed first reset for reproducibility
        state, info = env.reset(seed=args.seed if i_episode == 1 else None)
        unsafe_flag = False
        if args.verbosity > 1: print(i_episode, ": Real data")

        flags = []
        deviation = []
        while not done and not trunc:
            if safe_agent is not None:
                action, shielded, dev, policy_action = safe_agent(state)
                flags.append(shielded[0])
                deviation.append(dev)
            else:
                policy_action = agent(state)
                action = np.copy(policy_action)
                shielded = "N"

            next_state, reward, done, trunc, info = env.step(action)

            episode_steps += 1
            total_numsteps += 1
            episode_reward += reward


            cost = 0

            if env.unsafe(info['true_state'], False):

                real_unsafe_episodes += 1 * (not unsafe_flag)
                # episode_reward -= 100 * (not unsafe_flag)
                reward -= 100
                if args.verbosity > 1: print("UNSAFE (outside testing)", shielded)
                # print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                done = done or (True if safe_agent is not None else False)
                cost = 1

                unsafe_flag = True
            # Ignore the "done" signal if it comes from hitting the time
            # horizon.
            # github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py

            if cost > 0:
                agent.add(state, policy_action, reward, next_state, done or trunc, 1)
                real_data.push(state, action, reward, next_state, done or trunc, 1)
            else:
                agent.add(state, policy_action, reward, next_state, done or trunc, 0)
                real_data.push(state, action, reward, next_state, done or trunc, 0)

            if len(agent.memory) >= args.batch_size:

                losses = agent.train()
                if args.ppo:
                    writer.add_scalar(f'loss/policy_loss', losses['avg_policy_loss'], total_numsteps)
                    writer.add_scalar(f'loss/value_loss', losses['avg_value_loss'], total_numsteps)
                    writer.add_scalar(f'loss/entropy_loss', losses['avg_entropy_loss'], total_numsteps)
                    writer.add_scalar(f'loss/clip_fraction', losses['avg_clip_fraction'], total_numsteps)
                    writer.add_scalar(f'loss/kl_div', losses['avg_kl_divergence'], total_numsteps)
                    writer.add_scalar(f'loss/total_loss', losses['avg_total_loss'], total_numsteps)
                    writer.add_scalar(f'loss/explained_variance', losses['avg_explained_variance'], total_numsteps)
                else:
                    critic_1_loss, critic_2_loss, policy_loss, ent_l, alpha = losses

                    writer.add_scalar(f'loss/critic_1', critic_1_loss, total_numsteps)
                    writer.add_scalar(f'loss/critic_2', critic_2_loss, total_numsteps)
                    writer.add_scalar(f'loss/policy_loss', policy_loss, total_numsteps)
                    writer.add_scalar(f'loss/entropy_loss', ent_l, total_numsteps)
                    writer.add_scalar(f'loss/alpha', alpha, total_numsteps)


            state = next_state

        if args.verbosity > 0: print("Sequence", "".join(flags))
        if safe_agent is not None:
            try:
                s, a, b, t = safe_agent.report()
                if args.verbosity > 1:
                    print("Shield steps:", s, "  Neural steps:", a, "  Backup steps:", b)
                    print("Average time:", t / (s + a + b))
                safe_agent.reset_count()
            except Exception:
                pass

        total_real_episodes += 1
        avg_deviation.append([total_numsteps, np.mean(deviation) if len(deviation) > 0 else 0])
        writer.add_scalar(f'agent/avg_deviation', np.mean(deviation) if len(deviation) > 0 else 0, total_numsteps)


        if total_numsteps >= args.start_steps * train_steps and args.no_safety is False and args.dynamics is None:
        # if False:
            train_steps*=2
            try:

                states, actions, rewards, next_states, dones,  = \
                    real_data.sample(batch_size = min(len(real_data), 500000), get_cost=False, remove_samples=False, horizon = args.horizon)


            except Exception as e:
                print(traceback.format_exc())
                print("Error in sampling")
                exit()

            if env_model is not None:
                env_model.koopman_model.lr = 0.0003
                koopman_model = env_model.koopman_model
                epochs = 50
            else:
                koopman_model = None
                epochs = 200

            env_model, ev_score, r2_score, mean, std = get_environment_model(
                    states, actions, next_states, koopman_model = koopman_model, latent_dim=args.red_dim, horizon = args.horizon, epochs= epochs, percentile=args.percentile, adaptive_error=args.adaptive_error)

            writer.add_scalar(f'loss/ev_koopman', ev_score, total_numsteps)
            writer.add_scalar(f'loss/r2_score', r2_score, total_numsteps)
            safety_box = safety.Box(np.hstack([env.safety.lower, -np.ones((env.safety.lower.shape[0], args.red_dim))]), np.hstack([env.safety.upper, np.ones((env.safety.upper.shape[0], args.red_dim))]))

            if args.red_dim != 0:
                safety_box.lower[:, :-args.red_dim] = (safety_box.lower[:, :-args.red_dim] - mean)/(std + 1e-8)
                safety_box.upper[:, :-args.red_dim] = (safety_box.upper[:, :-args.red_dim] - mean)/(std + 1e-8)
            else:
                safety_box.lower = (safety_box.lower - mean)/(std + 1e-8)
                safety_box.upper = (safety_box.upper - mean)/(std + 1e-8)

            if args.red_dim != 0:
                new_obs_space = safety.Box(np.concatenate([np.nan_to_num(env.observation_space.low, nan=-9999, posinf=33333333, neginf=-33333333), -np.ones(args.red_dim, )]), np.concatenate([np.nan_to_num(env.observation_space.high, nan=-9999, posinf=33333333, neginf=-33333333), np.ones(args.red_dim, )]))

                new_obs_space.lower[:, :-args.red_dim] = (new_obs_space.lower[:, :-args.red_dim] - mean)/(std + 1e-8)
                new_obs_space.upper[:, :-args.red_dim] = (new_obs_space.upper[:, :-args.red_dim] - mean)/(std + 1e-8)

                new_obs_space = gym.spaces.Box(low=new_obs_space.lower[0], high=new_obs_space.upper[0], shape=(args.red_dim + env.observation_space.shape[0],))


            else:
                new_obs_space = gym.spaces.Box(low=np.nan_to_num(env.observation_space.low, nan=-9999, posinf=33333333, neginf=-33333333), high=np.nan_to_num(env.observation_space.high, nan=-9999, posinf=33333333, neginf=-33333333), shape=(env.observation_space.shape[0],))
                new_obs_space.low = (new_obs_space.low - mean)/(std + 1e-8)
                new_obs_space.high = (new_obs_space.high - mean)/(std + 1e-8)

            if args.verbosity > 2:
                print(new_obs_space)
                print(safety_box)

            polys = safety_box.to_hyperplanes(new_obs_space)
            unsafe_domains = safety_box.invert_polytope(new_obs_space)
            env.transformed_safe_polys = polys
            env.transformed_polys = unsafe_domains

            if not args.wp:
                if args.aspos:
                    shield = ASPOSCBFPolicy(
                        env_model, new_obs_space, env.observation_space,
                        env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.koopman_model.transform,
                        args.cbf_gamma, args.cbf_alpha)
                else:
                    shield = CBFPolicy(
                        env_model, new_obs_space, env.observation_space,
                        env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.koopman_model.transform,
                        args.cbf_gamma)
            else:
                shield = ProjectionPolicy(
                    env_model, new_obs_space,
                    env.action_space, args.horizon, env.transformed_polys, env.transformed_safe_polys, env_model.koopman_model.transform)

            safe_agent = Shield(shield, agent, mean, std)

            start_time = time.time()
            shield.update_model()
            end_time = time.time()
            if args.verbosity > 0: print(f"Precomputation Phase: {end_time - start_time}")
            if args.verbosity > 2: print(len(polys), "safe polys")

        # Test the agent periodically

        writer.add_scalar(f'reward/train', episode_reward, total_numsteps)
        if args.verbosity > 0: print(f"Episode: {i_episode}, total numsteps: {total_numsteps}, episode steps: {episode_steps}, reward: {round(episode_reward, 2)}")

        writer.add_scalar(f'agent/unsafe_real_episodes', real_unsafe_episodes, total_numsteps)
        writer.add_scalar(f'agent/unsafe_real_episodes_ratio', real_unsafe_episodes/total_real_episodes, total_numsteps)
        if safe_agent is not None:
            safe_agent.reset_count()

        if i_episode % 10 == 0:
            if args.verbosity > 0: print("starting testing...")
            avg_reward = 0.
            episodes = 1
            unsafe_episodes = 0
            avg_length = 0.
            shield_count = 0
            backup_count = 0
            neural_count = 0
            t = 0

            for episode_num in range(episodes):
                record_video = i_episode % 5 == 0  # Record every alternate episode (example condition)
                custom_filename = f"runs/{name}/videos/episode_{i_episode}.mp4"

                # video_env.video_recorder.file_prefix = os.path.join("videos/", f"{custom_filename.split('.')[0]}")

                state, info = env.reset()
                episode_reward = 0
                done = False
                trunc = False
                episode_steps = 0
                if args.render:
                    frames  = [env.render()]

                while not done and not trunc:
                    # Decide action
                    if safe_agent is not None:
                        action, shielded, dev, _ = safe_agent(state)
                    else:
                        action = agent(state)
                        shielded = None


                    next_state, reward, done, trunc, info = env.step(action)
                    episode_reward += reward
                    episode_steps += 1

                    if episode_steps >= env._max_episode_steps:
                        done = True
                    if env.unsafe(info['true_state'], False):
                        if args.verbosity > 2:
                            print("UNSAFE Inside testing", shielded)
                            # episode_reward += -100
                            print(f"{np.round(state, 2)}", "\n", action, "\n", f"{np.round(next_state, 2)}")
                        unsafe_episodes += 1
                        done = True

                    if done and safe_agent is not None:
                        try:
                            s, a, b, t = safe_agent.report()
                            if args.verbosity > 1: print("Finished test episode:", s, "shield and", b, "backup and", a, "neural")

                            shield_count += s
                            backup_count += b
                            neural_count += a

                            if args.verbosity > 1: print("Average time:", t / (s + a + b))
                            safe_agent.reset_count()
                        except Exception as e:
                            print(e)
                            pass

                    state = next_state
                    if args.render:
                        frames.append(env.render())

                if record_video and args.render:
                    imageio.mimsave(custom_filename, frames, fps=30)
                avg_reward += episode_reward
                avg_length += episode_steps

                avg_reward /= episodes
                avg_length /= episodes
                shield_count /= episodes
                neural_count /= episodes
                backup_count /= episodes
                unsafe_test_episodes+=unsafe_episodes
                total_test_episodes+=episodes
                writer.add_scalar(f'agent/shield', shield_count, total_numsteps)
                writer.add_scalar(f'agent/neural', neural_count, total_numsteps)
                writer.add_scalar(f'agent/backup', backup_count, total_numsteps)
                writer.add_scalar(f'agent/unsafe_test_episodes', unsafe_test_episodes, total_numsteps)
                writer.add_scalar(f'agent/unsafe_test_episodes_ratio', (unsafe_test_episodes+0.0000001)/(total_test_episodes + 0.0000001), total_numsteps)
                writer.add_scalar(f'reward/test', avg_reward, total_numsteps)

                if args.verbosity > 0:
                    print("----------------------------------------")
                    print("Test Episodes: {}, Unsafe: {}, Avg. Length: {}, Avg. Reward: {}"
                        .format(episodes, unsafe_episodes, round(avg_length, 2),
                                round(avg_reward, 2)))
                    print("----------------------------------------")
                # total_episodes += 1


        if total_numsteps > args.num_steps:
            break

    total_episodes = next(iterator_loop) - 1
    print("Total unsafe real:", real_unsafe_episodes, "/", total_real_episodes)
    print("Total unsafe real:", real_unsafe_episodes, "/", total_real_episodes, file=file)
    print("Total unsafe sim:", unsafe_sim_episodes, "/", total_sim_episodes)
    print("Total unsafe sim:", unsafe_sim_episodes, "/", total_sim_episodes, file=file)
    print("Total unsafe Test:", unsafe_test_episodes, "/", total_test_episodes)
    print("Total unsafe Test:", unsafe_test_episodes, "/", total_test_episodes, file=file)
    print("Using SPICE:", not args.no_safety)
    print("Using SPICE:", not args.no_safety, file=file)

    avg_deviation = np.array(avg_deviation)
    plt.figure()
    plt.plot(avg_deviation)
    plt.savefig(f"runs/{name}/deviation.png")
    plt.close()

    writer.add_hparams(
        hparam_dict = hyperparams,
        metric_dict = {
            "Unsafe Real Episodes": real_unsafe_episodes,
            "Unsafe Sim Episodes": unsafe_sim_episodes,
            "Unsafe Test Episodes": unsafe_test_episodes,
            "Total Real Episodes": total_real_episodes,
            "Total Sim Episodes": total_sim_episodes,
            "Total Test Episodes":total_test_episodes
        }
    )

    return { "real": {
        "unsafe": real_unsafe_episodes,
        "total": total_real_episodes
    },
    "sim": {
        "unsafe": unsafe_sim_episodes,
        "total": total_sim_episodes
    },
    "test": {
        "unsafe": unsafe_test_episodes,
        "total": total_test_episodes
    }
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='RAMPS Args')
    parser.add_argument('--env_name', default="lunar_lander")
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--seed', type=int, default=123456)
    parser.add_argument('--batch_size', type=int, default=2048)
    parser.add_argument('--mini_batch_size', type=int, default=256)
    parser.add_argument('--num_steps', type=int, default=200000)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--replay_size', type=int, default=5000000)
    parser.add_argument('--start_steps', type=int, default=10000)
    parser.add_argument('--cuda', action="store_true", default=False)
    parser.add_argument('--horizon', type=int, default=5)
    parser.add_argument('--cbf_gamma', type=float, default=0.7)
    parser.add_argument('--red_dim', type=int, default = 20)
    parser.add_argument('--no_safety', default=False, action='store_true')
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--percentile', default=99, type=int)
    parser.add_argument('--dynamics', default=None, type=str)
    parser.add_argument('--cbf_alpha', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.0)
    parser.add_argument('--verbosity', type=int, default=0)

    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                        help='target smoothing coefficient (tau) (default: 0.005)')
    parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                        help='Temperature parameter alpha determines the relative importance of the entropy\
                                term against the reward (default: 0.2)')
    parser.add_argument('--automatic_entropy_tuning', default=False, action='store_true',
                        help='Automaically adjust alpha (default: False)')
    parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                        help='model updates per simulator step (default: 1)')
    parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                        help='Value target update per no. of updates per step (default: 1)')

    parser.add_argument('--adaptive_error', default=False, action='store_true')
    parser.add_argument('--wp', default=False, action='store_true')
    parser.add_argument('--aspos', default=True, action='store_true')
    parser.add_argument('--ppo', default=True, action='store_true')
    parser.add_argument('--noise_type', default="uniform")

    args = parser.parse_args()

    run_experiment(args)
