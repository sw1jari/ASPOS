import glob
import statistics
from collections import defaultdict

# Collect the unsafe fraction of every run for real, sim and test episodes, as in predefined_exps.py
# Writes to logs/predefined/combind_results_{shield}.txt

# First we need sort the unsafe fractions of every run into groups by shield, env, agent, noise and kind
# so each group ends up with 5 fractions, one per seed
unsafe_fractions = defaultdict(list)
log_paths = glob.glob('logs/predefined/*/*/log.txt')
for log_path in log_paths:
    # print(f"Processing {log_path}")

    # Get shield, agent, env, and noise from the log path
    parts = log_path.split('/')
    shield = parts[2]
    run_name = parts[3]
    name_parts = run_name.split('_')
    agent = name_parts[2]
    env = name_parts[3]
    noise = name_parts[8][1:]
    # print(f"shield: {shield}, agent: {agent}, env: {env}, noise: {noise}")

    # Get the unsafe fractions for real, sim, and test episodes from the log file
    # The lines look like "Total unsafe real: 211 / 1305"
    for line in open(log_path):
        if not line.startswith('Total unsafe'):
            continue
        label, counts = line.strip().split(': ')
        kind = label.split(' ')[-1].lower()
        unsafe, total = counts.split(' / ')
        fraction = int(unsafe) / (int(total) + 1e-8)
        # print(f"  {kind}: {unsafe} / {total} = {fraction:.4f}")
        unsafe_fractions[shield, env, agent, noise, kind].append(fraction)

    # Break if you wanna debug using one file
    # break

# Then we wanna squash each group into its mean and std over the 5 seeds
for shield in ('ramps', 'aspos'):
    print(f"\n{shield}")
    for env in ('pendulum', 'cheetah', 'hopper', 'ant'):
        for agent in ('PPO', 'SAC'):
            for noise in ('0.1', '0.01', '0.001', '0.0'):
                print(f"{env.capitalize()} {agent} - Noise: {noise}")
                for kind in ('real', 'sim', 'test'):
                    fractions = unsafe_fractions[shield, env, agent, noise, kind]
                    mean = statistics.mean(fractions)
                    std = statistics.pstdev(fractions)
                    print(f"Total unsafe {kind}: {mean:.4f}  ± {std:.4f}")
                print()
