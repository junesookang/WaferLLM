set -e

P=$1
Mt=$(($2 / $1))
Kt=$(($3 / $1))
Nt=$(($4 / $1))
L=$5 # if 5 args are given, use the 5th as L, else default to 1
if [ -z "$L" ]; then
  L=1
fi

simulator=false

if [ -n "$5" ]; then
    simulator=$5
fi

python compile.py "$P" "$Mt" "$Kt" "$Nt" "$simulator" "$L"

if [ "$simulator" == "true" ]; then
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --L "$L" --simulator
else
    python launch_wse3.py --P "$1" --M "$2" --K "$3" --N "$4" --L "$L"
fi