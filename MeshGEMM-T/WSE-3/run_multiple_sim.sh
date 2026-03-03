#!/usr/bin/env bash
set -euo pipefail

SIM_RESULTS_FILE="sim_results.json"
declare -A existing_config_map=()

if [[ -f "$SIM_RESULTS_FILE" ]] && command -v python3 >/dev/null 2>&1; then
  mapfile -t existing_keys < <(python3 - <<'PY' "$SIM_RESULTS_FILE"
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])

try:
    text = path.read_text().strip()
except OSError:
    sys.exit(0)

if not text:
    sys.exit(0)

try:
    data = json.loads(text)
except json.JSONDecodeError as exc:
    print(f"Warning: failed to parse {path}: {exc}", file=sys.stderr)
    sys.exit(0)

if not isinstance(data, list):
    sys.exit(0)

for entry in data:
    if not isinstance(entry, dict):
        continue
    try:
        key = f"{int(entry['P'])}-{int(entry['M'])}-{int(entry['K'])}-{int(entry['L'])}-{int(entry['R'])}"
    except (KeyError, ValueError, TypeError):
        continue
    print(key)
PY
  )

  for config_key in "${existing_keys[@]}"; do
    [[ -z "$config_key" ]] && continue
    existing_config_map["$config_key"]=1
  done
fi

P_VALUES=(3 6 9)
M_FACTORS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16)
L_VALUES=(0 1 2)
R_VALUES=(0 1 2)

for P in "${P_VALUES[@]}"; do
  base_dim=$((P * 8))
  for factor in "${M_FACTORS[@]}"; do
    M=$((P * factor))
    K=2
    L=1
    R=1
    config_key="${P}-${M}-${K}-${L}-${R}"
    if [[ -v existing_config_map["$config_key"] ]]; then
      echo "Skipping run_sim.sh P=$P M=$M K=$K L=$L R=$R (already recorded in ${SIM_RESULTS_FILE})"
      continue
    fi

    echo "Running run_sim.sh P=$P M=$M K=$K L=$L R=$R"
    bash run_sim.sh "$P" "$M" "$K" "$L" "$R"
  done
done
