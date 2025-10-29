#!/usr/bin/env bash
set -euo pipefail

P_VALUES=(2 4 8)
K_FACTORS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20)
L_VALUES=(0 1 2 3)

for P in "${P_VALUES[@]}"; do
  base_dim=$((P * 7))
  for factor in "${K_FACTORS[@]}"; do
    M=$((P * factor))
    K=$base_dim
    N=$base_dim
    for L in "${L_VALUES[@]}"; do
      echo "Running run_sim.sh P=$P M=$M K=$K N=$N L=$L"
      bash run_sim.sh "$P" "$M" "$K" "$N" "$L"
    done
  done
done
