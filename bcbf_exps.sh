#!/bin/sh
# BCBF baseline (Vahs, Pek and Tumova 2023) on the same grid as predefined_exps.py
# Run from the repo root with the venv active, outputs go to logs/bcbf/<noise type>
export ROOT="$(pwd)"
export MPLBACKEND=Agg

# Runs in this order, put the noise type you need most first (i think uniform is better to run first)
NOISE_TYPES="uniform gaussian"
# You can increase this if you have enough CPU cores / RAM
NUM_PARALLEL_RUNS=7

for noise_type in $NOISE_TYPES; do
    for noise in 0.1 0.01 0.001 0.0; do
        # Noise 0 is the same run for both types, so it only runs under uniform
        [ "$noise_type" = gaussian ] && [ "$noise" = 0.0 ] && continue
        for env in pendulum cheetah hopper ant; do
            for seed in 1 3 5 11 42; do
                echo "$noise_type $env $noise $seed"
            done
        done
    done
done | xargs -P "$NUM_PARALLEL_RUNS" -L 1 sh -c '
    mkdir -p "$ROOT/logs/bcbf/$1" && cd "$ROOT/logs/bcbf/$1"
    python "$ROOT/combined_main.py" --bcbf --noise_type $1 --env_name $2 --noise $3 --seed $4 --red_dim 0 --batch_size 2048 --replay_size 1000000 --cuda > bcbf_$2_n$3_s$4.log 2>&1
' sh
