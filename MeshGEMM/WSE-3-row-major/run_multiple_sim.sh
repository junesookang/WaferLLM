#!/usr/bin/env bash
set -euo pipefail

P_VALUES=(3 6 9)
M_FACTORS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
L_VALUES=(0 1 2)

for P in "${P_VALUES[@]}"; do
  base_dim=$((P * 8))
  for factor in "${M_FACTORS[@]}"; do
    M=$((P * factor))
    K=$base_dim
    N=$base_dim
    for L in "${L_VALUES[@]}"; do
      echo "Running run_sim.sh P=$P M=$M K=$K N=$N L=$L"
      bash run_sim.sh "$P" "$M" "$K" "$N" "$L"
    done
    rm -rf wio_flows_tmpdir.*
  done
done
