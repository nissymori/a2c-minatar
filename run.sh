#! /bin/bash

for ent_coef in 0.0 0.01 0.03 0.1 0.3; do
for seed in $(seq 1); do
for game in "asterix" "breakout" "freeway" "seaquest" "space_invaders"; do
    python3 -O train.py game=$game ent_coef=$ent_coef seed=$seed minatar_version=v0 &
done
wait
done
done
