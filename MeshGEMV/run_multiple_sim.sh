#!/usr/bin/env bash
set -euo pipefail

# Optional arguments allow quick sweeps of configurations without editing the file:
#   $0 [M_eq_N=true|false] [group_num]
if [[ "$M_eq_N" != "true" && "$M_eq_N" != "false" ]]; then
  echo "Usage: $0 [true|false] [group_num]" >&2
  exit 1
fi

GROUP_NUM=${2:-2}
if ! [[ "$GROUP_NUM" =~ ^[0-9]+$ ]] || [[ "$GROUP_NUM" -lt 1 ]]; then
  echo "Error: group_num must be a positive integer" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_DIR="${SCRIPT_DIR}/WSE-3"

if [[ ! -d "$TARGET_DIR" ]]; then
  echo "Error: expected directory ${TARGET_DIR} to exist" >&2
  exit 1
fi

trap 'popd >/dev/null' EXIT
pushd "$TARGET_DIR" >/dev/null

SIM_RESULTS_FILE="sim_results.json"
declare -A existing_config_map=()

if [[ -f "$SIM_RESULTS_FILE" ]] && command -v python3 >/dev/null 2>&1; then
  mapfile -t existing_keys < <(python3 - "$SIM_RESULTS_FILE" <<'PY'
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
        key = f"{int(entry['P'])}-{int(entry['M'])}-{int(entry['N'])}-{int(entry['L'])}"
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

P_VALUES=(8 16)
M_FACTORS=(1 2 4)
N_FACTORS=(1 2 4)
L_VALUES=(1 2)

for P in "${P_VALUES[@]}"; do
  if (( P % GROUP_NUM != 0 )); then
    echo "Skipping P=$P because group_num=$GROUP_NUM does not divide P"
    continue
  fi

  for mt_factor in "${M_FACTORS[@]}"; do
    M=$((P * mt_factor))

    if [[ "$M_eq_N" == "true" ]]; then
      N=$M
      for L in "${L_VALUES[@]}"; do
        config_key="${P}-${M}-${N}-${L}"
        if [[ -v existing_config_map["$config_key"] ]]; then
          echo "Skipping run_sim.sh P=$P M=$M N=$N L=$L (already recorded in ${SIM_RESULTS_FILE})"
          continue
        fi
        echo "Running run_sim.sh P=$P M=$M N=$N L=$L group_num=$GROUP_NUM"
        bash run_sim.sh "$P" "$M" "$N" "$L" "$GROUP_NUM"
        existing_config_map["$config_key"]=1
        rm -rf wio_flows_tmpdir.*
      done
      continue
    fi

    for nt_factor in "${N_FACTORS[@]}"; do
      N=$((P * nt_factor))
      for L in "${L_VALUES[@]}"; do
        config_key="${P}-${M}-${N}-${L}"
        if [[ -v existing_config_map["$config_key"] ]]; then
          echo "Skipping run_sim.sh P=$P M=$M N=$N L=$L (already recorded in ${SIM_RESULTS_FILE})"
          continue
        fi
        echo "Running run_sim.sh P=$P M=$M N=$N L=$L group_num=$GROUP_NUM"
        bash run_sim.sh "$P" "$M" "$N" "$L" "$GROUP_NUM"
        existing_config_map["$config_key"]=1
        rm -rf wio_flows_tmpdir.*
      done
    done
  done
done
