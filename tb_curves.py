import csv
import glob
import math
import statistics
from multiprocessing import Pool

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

# Collect the training curves of every run, the episodic reward and the cumulative unsafe episodes
# Same curves as figure 1 in RAMPS
# Writes them every 5000 env steps to logs/predefined/curves.csv, so the seeds line up for mean and std

OUT = 'logs/predefined/curves.csv'
WINDOW = 5000
STEPS = range(WINDOW, 200000 + 1, WINDOW)


def read_curves(tb_path):
    # Get shield, agent, env, noise and seed from the tb path, same as in tb_stats
    parts = tb_path.split('/')
    shield = parts[2]
    run_name = parts[3]
    name_parts = run_name.split('_')
    agent = name_parts[2]
    env = name_parts[3]
    seed = name_parts[7][1:]
    noise = name_parts[8][1:]

    # Read the tb file, it has one value per training episode, logged at the env step where the episode ended
    acc = EventAccumulator(tb_path)
    acc.Reload()
    rewards = acc.Scalars('reward/train')
    unsafe_counts = acc.Scalars('agent/unsafe_real_episodes')

    # At every grid step, the mean reward of the episodes that ended in the window before it
    # and the number of unsafe episodes so far
    rows = []
    for step in STEPS:
        window_rewards = [e.value for e in rewards if step - WINDOW < e.step <= step]
        counts_so_far = [e.value for e in unsafe_counts if e.step <= step]
        rows.append({
            'shield': shield,
            'env': env,
            'agent': agent,
            'noise': noise,
            'seed': seed,
            'step': step,
            'reward': statistics.mean(window_rewards) if window_rewards else float('nan'),
            'unsafe_episodes': counts_so_far[-1] if counts_so_far else 0,
        })
    return rows


# Only the main process runs this part, the 5 workers only run read_curves
if __name__ == '__main__':
    tb_paths = glob.glob('logs/predefined/*/*/events.out.tfevents.*')
    print(f"Found {len(tb_paths)} tb files")

    # Read the runs 5 at a time, because each SAC file takes about 16 s
    with Pool(5) as pool:
        runs = list(tqdm(pool.imap_unordered(read_curves, tb_paths), total=len(tb_paths)))

    # Sorted so the file comes out the same every time, the steps of a run stay in order
    rows = []
    for run in runs:
        rows += run
    rows = sorted(rows, key=lambda row: (row['shield'], row['env'], row['agent'], row['noise'], int(row['seed'])))
    with open(OUT, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    missing = sum(math.isnan(row['reward']) for row in rows)
    print(f"wrote {len(runs)} runs of {len(STEPS)} steps to {OUT}, {missing} windows without an episode")
