import csv
import glob
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import style

# Main result, ratio of unsafe training episodes (top) and ratio of backup test steps (bottom) against the noise level
# Both shields and both agents per env, mean and std over the 5 seeds
# Writes logs/plots/noise.pdf with the legend above the panels
# and logs/plots/noise_legend_inside.pdf with the legend in the top left panel

OUT = 'logs/plots/noise.pdf'
OUT_LEGEND_INSIDE = 'logs/plots/noise_legend_inside.pdf'

# Unsafe fraction of the training episodes of every run, from its log.txt
unsafe = defaultdict(list)
for log_path in glob.glob('logs/predefined/*/*/log.txt'):
    parts = log_path.split('/')
    shield = parts[2]
    name_parts = parts[3].split('_')
    agent = name_parts[2]
    env = name_parts[3]
    noise = name_parts[8][1:]
    for line in open(log_path):
        if line.startswith('Total unsafe real'):
            unsafe_episodes, total_episodes = line.split(': ')[1].split(' / ')
            unsafe[shield, env, agent, noise].append(int(unsafe_episodes) / int(total_episodes))

# Backup ratio of the test episodes of every run, from tb_stats
backup = defaultdict(list)
for row in csv.DictReader(open('logs/predefined/actions.csv')):
    backup[row['shield'], row['env'], row['agent'], row['noise']].append(float(row['backup_ratio']))

# One panel per env and quantity, one line per shield and agent
fig, axes = plt.subplots(2, len(style.ENVS), figsize=(style.FULL_WIDTH, 2.6), sharex=True, sharey=True, layout='constrained')
x = np.arange(len(style.NOISES))
for panel_row, (values, ylabel) in enumerate([(unsafe, 'unsafe ratio'), (backup, 'backup ratio')]):
    for panel_col, env in enumerate(style.ENVS):
        ax = axes[panel_row, panel_col]
        for shield in style.SHIELDS:
            for agent in style.AGENTS:
                runs = [values[shield, env, agent, noise] for noise in style.NOISES]
                mean = np.array([np.mean(r) for r in runs])
                std = np.array([np.std(r) for r in runs])
                color = style.SHIELD_COLORS[shield]
                ax.plot(x, mean, color=color, linestyle=style.AGENT_LINES[agent], marker='o', markersize=3, label=f'{style.SHIELD_NAMES[shield]} {agent}')
                ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x, style.NOISES)
        if panel_row == 0:
            ax.set_title(style.ENV_NAMES[env])
        if panel_row == 1:
            ax.set_xlabel(r'noise $\alpha$')
        if panel_col == 0:
            ax.set_ylabel(ylabel)

# One legend for the whole figure, above the panels
handles, labels = axes[0, 0].get_legend_handles_labels()
legend = fig.legend(handles, labels, loc='outside upper center', ncol=4)
os.makedirs('logs/plots', exist_ok=True)
fig.savefig(OUT)
print(f"wrote {OUT}")

# Same figure with the legend in the top left panel, pendulum stays near 0 there until the last noise level
legend.remove()
axes[0, 0].legend(loc='upper left')
fig.savefig(OUT_LEGEND_INSIDE)
print(f"wrote {OUT_LEGEND_INSIDE}")
