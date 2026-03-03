#!/usr/bin/env bash
set -euo pipefail

M_eq_N=${1:-false}

P_VALUES=(6)
M_FACTORS=(1 2 3 4 5 6 7 8 9)
L_VALUES=(0 1)

for P in "${P_VALUES[@]}"; do
  base_dim=$((P * 8))
  for factor in "${M_FACTORS[@]}"; do
    M=$((P * factor))
    K=$base_dim
    if [ "$M_eq_N" = true ]; then
      N=$M
    else
      N=$base_dim
    fi
    for L in "${L_VALUES[@]}"; do
      echo "Running run_sim.sh P=$P M=$M K=$K N=$N L=$L"
      bash run_sim.sh "$P" "$M" "$K" "$N" "$L"
    done
    rm -rf wio_flows_tmpdir.*
  done
done
