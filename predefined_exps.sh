#!/bin/sh
mkdir -p logs/predefined/runs logs/predefined/runs_sac
cd logs/predefined
export MPLBACKEND=Agg

for env in pendulum cheetah hopper ant; do
    for method in sac ppo; do
        for noise in 0.1 0.01 0.001 0.0; do
            for seed in 1 3 5 11 42; do
                echo "$method $env $noise $seed"
            done
        done
    done
done | xargs -P 6 -L 1 sh -c '
    extra=""
    [ "$1" = ppo ] && extra="--batch_size 2048"
    python ../../main_$1.py --env_name $2 --noise $3 --cbf_alpha 0 --seed $4 --red_dim 0 --num_steps 200000 --cuda $extra > $1_$2_n$3_s$4.log 2>&1
' sh
