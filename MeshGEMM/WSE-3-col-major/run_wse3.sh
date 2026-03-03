set -e

P=$1
Mt=$(($2 / $1))
if [ $(($2 % $1)) -ne 0 ]; then
  Mt=$((Mt + 1))
fi
Kt=$(($3 / $1))
if [ $(($3 % $1)) -ne 0 ]; then
  Kt=$((Kt + 1))
fi
Nt=$(($4 / $1))
if [ $(($4 % $1)) -ne 0 ]; then
  Nt=$((Nt + 1))
fi
L=$6 # if 6 args are given, use the 6th as L, else default to 1
if [ -z "$L" ]; then
  L=1
fi
C=$7 # if 7 args are given, use the 7th as C, else default to 1
if [ -z "$C" ]; then
  C=1
fi

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

python compile.py "$P" "$Mt" "$Kt" "$Nt" "$simulator" "$L"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --L "$L" --C "$C" --simulator
else
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --L "$L" --C "$C"
fi