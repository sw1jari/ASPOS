import csv
import os
from collections import defaultdict

import numpy as np

import style

# Backup share of our RAMPS runs without noise
# Only the backup share, in every other step the QP was solved
# The neural and shield split is left out, the 1e-8 check in src/policy.py counts solver rounding as shielding
# Writes logs/tables/backup.tex

OUT = 'logs/tables/backup.tex'

# Backup share of our RAMPS runs without noise in percent, one value per seed
backup = defaultdict(list)
for row in csv.DictReader(open('logs/predefined/actions.csv')):
    if row['shield'] == 'ramps' and row['noise'] == '0.0':
        backup[row['env'], row['agent']].append(100 * float(row['backup_ratio']))

# One table row per env, mean and std over the seeds
rows = []
for env in style.ENVS:
    cells = [style.ENV_NAMES[env]]
    for agent in style.AGENTS:
        mean = np.mean(backup[env, agent])
        std = np.std(backup[env, agent])
        cells.append(f'{mean:.1f} $\\pm$ {std:.1f}')
    rows.append(' & '.join(cells) + r' \\')

caption = (r'Share of test steps in which the backup policy acted, in percent, without noise ($\alpha = 0$), '
           r'mean $\pm$ std over 5 seeds. In all other steps the QP of the shield was solved.')
table = [
    r'\begin{table}[t]',
    r'\centering',
    rf'\caption{{{caption}}}',
    r'\label{tab:backup}',
    r'\begin{tabular}{lrr}',
    r'\toprule',
    r'Env & SAC & PPO \\',
    r'\midrule',
    *rows,
    r'\bottomrule',
    r'\end{tabular}',
    r'\end{table}',
]

os.makedirs('logs/tables', exist_ok=True)
open(OUT, 'w').write('\n'.join(table) + '\n')
print(f"wrote {OUT}")
