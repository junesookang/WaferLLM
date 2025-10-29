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
Nt=$(($4 / $1))
if [ $(($4 % $1)) -ne 0 ]; then
  Nt=$((Nt + 1))
fi
L=$5 # if 5 args are given, use the 5th as L, else default to 1
if [ -z "$L" ]; then
  L=1
fi

echo "P=$1, M=$2, K=$3, N=$4, Mt=$Mt, Kt=$Kt, Nt=$Nt, L=$L"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P:"$1",Mt:"$Mt",Kt:"$Kt",Nt:"$Nt",L:"$L" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P "$1" --M "$2" --K "$3" --N "$4" --L "$L"
