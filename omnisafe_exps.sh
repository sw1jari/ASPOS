#!/bin/sh
# Runs the OmniSafe baselines on the grid of predefined_exps.py, NUM_PARALLEL_RUNS at a time
# Run from the repo root with the omnisafe env active, the outputs go to logs/omnisafe
# Run this script with sh omnisafe_exps.sh or bash, not zsh, which keeps $ALGOS etc as one word
mkdir -p logs/omnisafe
cd logs/omnisafe

# Each run uses one CPU thread and well under 1 GB of memory, so you can run a lot in parallel
NUM_PARALLEL_RUNS=8

# You can select multiple or one of: PPOLag, CUP, P3O.
ALGOS="PPOLag CUP P3O"

NOISES="0.1 0.0 0.01 0.001"

ENVS="pendulum cheetah hopper ant"

SEEDS="1 3 5 11 42"

# One line per run: log name, then the arguments, noise 0.1 and 0 first
runs() {
    for noise in $NOISES; do
        for algo in $ALGOS; do
            for env in $ENVS; do
                for seed in $SEEDS; do
                    echo "${algo}_${env}_n${noise}_s$seed --algo $algo --env_name $env --noise $noise --seed $seed"
                done
            done
        done
    done
}

# Keep NUM_PARALLEL_RUNS runs going until the list is empty
runs | xargs -P "$NUM_PARALLEL_RUNS" -L 1 sh -c 'name=$1; shift; python ../../main_omnisafe.py "$@" > "$name.log" 2>&1' sh
