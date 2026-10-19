import csv
import glob
import statistics
from collections import defaultdict
from multiprocessing import Pool

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

# Collect who picked the action in the test episodes of every run, the agent (neural), the shield or the backup
# Same idea as the action type table in RAMPS (appendix A.5.5, table 3)
# Writes one row per run to logs/predefined/actions.csv, tables and plots average it later

OUT = 'logs/predefined/actions.csv'
NUM_WORKERS = 8
NOISE_LEVELS = ['0.0', '0.001', '0.01', '0.1']


def read_ratios(tb_path):
    # Get shield, agent, env, noise and seed from the tb path, same as in log_stats
    parts = tb_path.split('/')
    shield = parts[2]
    run_name = parts[3]
    name_parts = run_name.split('_')
    agent = name_parts[2]
    env = name_parts[3]
    seed = name_parts[7][1:]
    noise = name_parts[8][1:]
    # print(f"shield: {shield}, agent: {agent}, env: {env}, noise: {noise}, seed: {seed}")

    # Read the tb file, it has one value per test episode, the steps where neural, shield or backup picked the action
    acc = EventAccumulator(tb_path)

    # This reads the file. For some reason it's called Reload().
    acc.Reload()
    neural_steps = [e.value for e in acc.Scalars('agent/neural')]
    shield_steps = [e.value for e in acc.Scalars('agent/shield')]
    backup_steps = [e.value for e in acc.Scalars('agent/backup')]

    # Per test episode the share of its steps each one picked, skipping the episodes from before the shield exists
    neural_ratios, shield_ratios, backup_ratios = [], [], []
    for neural_step, shield_step, backup_step in zip(neural_steps, shield_steps, backup_steps):
        length = neural_step + shield_step + backup_step
        if length == 0:
            continue
        neural_ratios.append(neural_step / length)
        shield_ratios.append(shield_step / length)
        backup_ratios.append(backup_step / length)
        # print(f"length: {length:.0f}, neural: {neural_step / length:.3f}, shield: {shield_step / length:.3f}, backup: {backup_step / length:.3f}")

    # Average the per episode shares so every episode counts the same, like RAMPS table 3
    return {
        'shield': shield,
        'env': env,
        'agent': agent,
        'noise': noise,
        'seed': seed,
        'test_episodes': len(backup_steps),
        'shielded_episodes': len(backup_ratios),
        'neural_ratio': statistics.mean(neural_ratios),
        'shield_ratio': statistics.mean(shield_ratios),
        'backup_ratio': statistics.mean(backup_ratios),
        'run': run_name,
    }


# Only the main process runs this part, the 5 workers only run read_ratios
if __name__ == '__main__':
    # First we need the tb file of every run, there is exactly one per run folder
    tb_paths = glob.glob('logs/predefined/*/*/events.out.tfevents.*')
    print(f"Found {len(tb_paths)} tb files")

    # Uncomment if you wanna debug using one file
    # tb_paths = tb_paths[:1]

    # Read multiple tb files in parallel bc it is slow
    with Pool(NUM_WORKERS) as pool:
        rows = list(tqdm(pool.imap_unordered(read_ratios, tb_paths), total=len(tb_paths)))

    # Print the mean backup ratio over the seeds of every setting, so we can eyeball the results
    backup_ratios = defaultdict(list)
    for row in rows:
        backup_ratios[row['shield'], row['env'], row['agent'], row['noise']].append(row['backup_ratio'])
    print(f"\nmean backup ratio for noise {' '.join(NOISE_LEVELS)}")
    for shield in ('ramps', 'aspos'):
        for env in ('pendulum', 'cheetah', 'hopper', 'ant'):
            for agent in ('PPO', 'SAC'):
                means = [statistics.mean(backup_ratios[shield, env, agent, noise]) for noise in NOISE_LEVELS]
                print(f"{shield:6} {env:9} {agent:4}" + ''.join(f"{mean:7.3f}" for mean in means))

    # Sorted by shield and run folder so the file comes out the same every time
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row['shield'], row['run'])))
    print(f"wrote {len(rows)} runs to {OUT}")
