set -e
fabric_w=$(($1 + 7))
fabric_h=$(($1 + 2))

Mt=$(($2 / $1))
if [ $(($2 % $1)) -ne 0 ]; then
  Mt=$((Mt + 1))
fi
Kt=$(($3 / $1))
if [ $(($3 % $1)) -ne 0 ]; then
  Kt=$((Kt + 1))
fi
L=$4 # if 4 args are given, use the 4th as L, else default to 1
if [ -z "$L" ]; then
  L=1
fi
R=$5 # if 5 args are given, use the 5th as R, else default to 1
if [ -z "$R" ]; then
  R=1
fi

echo "P=$1, M=$2, K=$3, Mt=$Mt, Kt=$Kt, L=$L, R=$R"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P:"$1",Mt:"$Mt",Kt:"$Kt",L:"$L",R:"$R" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P "$1" --M "$2" --K "$3" --L "$L" --R "$R"

rm -rf wio_flows_tmpdir.*