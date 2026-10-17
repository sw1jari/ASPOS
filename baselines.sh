#!/bin/sh
# Run from the repo root with the venv active, the outputs go to logs/baselines
mkdir -p logs/baselines
cd logs/baselines

# Tk crashed a background run before, plot without a display
export MPLBACKEND=Agg

NOISE=0.1
SEEDS="0 1 2"
NUM_STEPS_CHEETAH=300000
NUM_STEPS_HOPPER=300000
NUM_STEPS_PENDULUM=200000
HORIZON=5
RED_DIM=0
CBF_GAMMA=0.7
CBF_ALPHA=0
NUM_PARALLEL_RUNS=5

runs() {
    printf '%s\n' "cheetah $NUM_STEPS_CHEETAH" "hopper $NUM_STEPS_HOPPER" "pendulum $NUM_STEPS_PENDULUM" | while read -r env steps; do
        for seed in $SEEDS; do
            flags="--env_name $env --num_steps $steps --seed $seed --horizon $HORIZON --red_dim $RED_DIM --cbf_gamma $CBF_GAMMA --cbf_alpha $CBF_ALPHA"
            echo "ramps_n0_${env}_s$seed $flags --noise 0"
            echo "ramps_n${NOISE}_${env}_s$seed $flags --noise $NOISE"
            echo "sac_n${NOISE}_${env}_s$seed $flags --noise $NOISE --no_safety"
        done
    done
}

# Keep NUM_PARALLEL_RUNS runs going until the list is empty
runs | xargs -P "$NUM_PARALLEL_RUNS" -L 1 sh -c 'name=$1; shift; python ../../main_sac.py "$@" > "$name.log" 2>&1' sh
