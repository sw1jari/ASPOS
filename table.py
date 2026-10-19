import glob
import os
import statistics
import sys
from collections import defaultdict

ENVS = {'pendulum': 'Pendulum', 'cheetah': 'Cheetah', 'hopper': 'Hopper', 'ant': 'Ant'}
AGENTS = ['sac', 'ppo']
NOISES = ['0.0', '0.001', '0.01', '0.1']
SHIELDS = {'RAMPS': 'logs/predefined/ramps', 'ASPOS': 'logs/predefined/aspos'}
OUT = sys.argv[1] if len(sys.argv) > 1 else 'logs/tables'
CAPTION = (r'Unsafe training episodes under uniform observation noise $U(-\alpha, \alpha)$, mean $\pm$ std over 5 seeds. '
           r'RAMPS is the original shield, whose tightening only covers model error. ASPOS adds the noise tightening; at $\alpha = 0$ the two are identical.')


def stats(folder):
    # Mean and std of the final unsafe episode count per env, agent and noise
    counts = defaultdict(list)
    for log_path in glob.glob(f'{folder}/*/log.txt'):
        # Agent, env and noise from the run folder name, same as in log_stats
        name_parts = log_path.split('/')[3].split('_')
        agent = name_parts[2].lower()
        env = name_parts[3]
        noise = name_parts[8][1:]

        # The unsafe count from the line "Total unsafe real: 211 / 1305"
        for line in open(log_path):
            if line.startswith('Total unsafe real'):
                unsafe_episodes = int(line.split(': ')[1].split(' / ')[0])
                counts[env, agent, noise].append(unsafe_episodes)
    return {k: (round(statistics.mean(v)), round(statistics.pstdev(v))) for k, v in counts.items()}


results = {shield: stats(folder) for shield, folder in SHIELDS.items()}
ROWS = [(e, a) for e in ENVS for a in AGENTS]


def column(shields, noise):
    # Phantom zeros pad every number to the widest in its column, so the plus-minus signs line up
    found = {(s, e, a): results[s].get((e, a, noise)) for s in shields for e, a in ROWS}
    wm = max((len(str(r[0])) for r in found.values() if r), default=1)
    ws = max((len(str(r[1])) for r in found.values() if r), default=1)
    pad = lambda x, w: (f'\\phantom{{{"0" * (w - len(str(x)))}}}' if len(str(x)) < w else '') + str(x)
    return {k: f'${pad(r[0], wm)} \\pm {pad(r[1], ws)}$' if r else r'\multicolumn{1}{c}{--}' for k, r in found.items()}


def table(lines, spec, label):
    return '\n'.join([r'\begin{table*}[t]', r'\centering', rf'\caption{{{CAPTION}}}', rf'\label{{{label}}}',
                      rf'\begin{{tabular}}{{{spec}}}', r'\toprule', *lines, r'\bottomrule', r'\end{tabular}', r'\end{table*}', ''])


def separate(lines, e, a):
    if a == AGENTS[-1] and e != list(ENVS)[-1]:
        lines.append(r'\midrule')


# Version 1, noise across, RAMPS and ASPOS as sub rows
cols = {n: column(SHIELDS, n) for n in NOISES}
v1 = ['Env & Agent & Shield & ' + ' & '.join(f'$\\alpha = {float(n):g}$' for n in NOISES) + r' \\', r'\midrule']
for e, a in ROWS:
    for s in SHIELDS:
        first = s == 'RAMPS'
        v1.append(f'{ENVS[e] if first and a == AGENTS[0] else ""} & {a.upper() if first else ""} & {s} & '
                  + ' & '.join(cols[n][s, e, a] for n in NOISES) + r' \\')
    separate(v1, e, a)

# Version 2, RAMPS and ASPOS side by side per noise level, one shared column for alpha = 0
cols[NOISES[0]] = column(['RAMPS'], NOISES[0])
v2 = [r'& & & ' + ' & '.join(f'\\multicolumn{{2}}{{c}}{{$\\alpha = {float(n):g}$}}' for n in NOISES[1:]) + r' \\',
      r'\cmidrule(lr){4-5} \cmidrule(lr){6-7} \cmidrule(lr){8-9}',
      r'Env & Agent & $\alpha = 0$ & ' + ' & '.join(['RAMPS & ASPOS'] * 3) + r' \\', r'\midrule']
for e, a in ROWS:
    cells = [cols[NOISES[0]]['RAMPS', e, a]] + [cols[n][s, e, a] for n in NOISES[1:] for s in SHIELDS]
    v2.append(f'{ENVS[e] if a == AGENTS[0] else ""} & {a.upper()} & ' + ' & '.join(cells) + r' \\')
    separate(v2, e, a)

os.makedirs(OUT, exist_ok=True)
open(f'{OUT}/noise_v1.tex', 'w').write(table(v1, 'lllrrrr', 'tab:noise-v1'))
open(f'{OUT}/noise_v2.tex', 'w').write(table(v2, 'llrrrrrrr', 'tab:noise-v2'))
print(f'wrote {OUT}/noise_v1.tex and {OUT}/noise_v2.tex')
