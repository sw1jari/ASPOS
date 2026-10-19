import matplotlib.pyplot as plt

# Shared look of all plots and tables, import it before plotting
# Figures are made at their printed size, so the font size is the size in the paper
# 7 pt because the default font has taller letters than the 9 pt Libertine of the paper text
plt.rcParams['font.size'] = 7

# Printed widths in inches in the ACM sigconf paper
FULL_WIDTH = 7.0
COLUMN_WIDTH = 3.33

# Order of shields, envs, agents and noise levels everywhere
SHIELDS = ['ramps', 'aspos']
ENVS = ['pendulum', 'cheetah', 'hopper', 'ant']
AGENTS = ['SAC', 'PPO']
NOISES = ['0.0', '0.001', '0.01', '0.1']

# How each shield, agent and noise level is drawn
SHIELD_COLORS = {'ramps': 'C0', 'aspos': 'C1'}
AGENT_LINES = {'SAC': '-', 'PPO': '--'}
NOISE_COLORS = {'0.0': 'C0', '0.001': 'C1', '0.01': 'C2', '0.1': 'C3'}

# Names in the paper
SHIELD_NAMES = {'ramps': 'RAMPS', 'aspos': 'ASPOS'}
ENV_NAMES = {'pendulum': 'Pendulum', 'cheetah': 'Cheetah', 'hopper': 'Hopper', 'ant': 'Ant'}
