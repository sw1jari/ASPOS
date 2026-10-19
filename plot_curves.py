import csv
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

import style

# Training curves like figure 1 in RAMPS, cumulative unsafe episodes (top) and episode reward (bottom) against env steps
# One line per noise level, mean and std over the 5 seeds
# Writes one figure per shield and agent with all envs to logs/plots/curves_{shield}_{agent}.pdf

# Reward and unsafe episodes of every run as lists over the steps, curves.csv has the steps of a run in order
rows = list(csv.DictReader(open('logs/predefined/curves.csv')))
steps = np.array(sorted({int(row['step']) for row in rows}))
reward = defaultdict(list)
unsafe = defaultdict(list)
for row in rows:
    run = (row['shield'], row['env'], row['agent'], row['noise'], row['seed'])
    reward[run].append(float(row['reward']))
    unsafe[run].append(float(row['unsafe_episodes']))


def seeds_of(curves, shield, env, agent, noise):
    # The curves of all seeds of one setting, as an array of seeds by steps
    setting = (shield, env, agent, noise)
    return np.array([curve for run, curve in curves.items() if run[:4] == setting])


def draw(top_ax, bottom_ax, shield, env, agent):
    # One line per noise level, unsafe episodes on top and reward below
    for noise in style.NOISES:
        color = style.NOISE_COLORS[noise]
        for ax, curves in [(top_ax, unsafe), (bottom_ax, reward)]:
            runs = seeds_of(curves, shield, env, agent, noise)
            mean = runs.mean(axis=0)
            std = runs.std(axis=0)
            ax.plot(steps / 1000, mean, color=color, label=rf'$\alpha$ = {noise}')
            ax.fill_between(steps / 1000, mean - std, mean + std, color=color, alpha=0.15)
    bottom_ax.set_xlabel('env steps (thousands)')


os.makedirs('logs/plots', exist_ok=True)

# One figure per shield and agent with all envs, for the appendix
for shield in style.SHIELDS:
    for agent in style.AGENTS:
        fig, axes = plt.subplots(2, len(style.ENVS), figsize=(style.FULL_WIDTH, 2.6), sharex=True, layout='constrained')
        for col, env in enumerate(style.ENVS):
            draw(axes[0, col], axes[1, col], shield, env, agent)
            axes[0, col].set_title(style.ENV_NAMES[env])
        axes[0, 0].set_ylabel('unsafe episodes')
        axes[1, 0].set_ylabel('episode reward')
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='outside upper center', ncol=4)
        out = f'logs/plots/curves_{shield}_{agent}.pdf'
        fig.savefig(out)
        print(f"wrote {out}")
