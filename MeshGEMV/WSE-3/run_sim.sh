set -e
fabric_w=$(($1 + 7))
fabric_h=$(($1 + 2))

Mt=$(($2 / $1))
if [ $(($2 % $1)) -ne 0 ]; then
  Mt=$((Mt + 1))
fi
Nt=$(($3 / $1))
if [ $(($3 % $1)) -ne 0 ]; then
  Nt=$((Nt + 1))
fi
L=$4
if [ -z "$L" ]; then
  L=1
fi

pe_num_group=$(($1 / $5))
root_1st_phase=$((pe_num_group / 2))
root_2nd_phase=$(((($5 / 2) * pe_num_group) + root_1st_phase))

echo "P=$1, M=$2, N=$3, L=$4, group_num=$5, pe_num_group=$pe_num_group, root_1st_phase=$root_1st_phase, root_2nd_phase=$root_2nd_phase"

cslc --arch=wse3 ./src/layout.csl --fabric-dims="$fabric_w","$fabric_h" --fabric-offsets=4,1 \
    --params=P:"$1",Mt:"$Mt",Nt:"$Nt",L:"$L",pe_num_group:"$pe_num_group",root_1st_phase:"$root_1st_phase",root_2nd_phase:"$root_2nd_phase" \
    -o out --memcpy --channels 1

cs_python ./launch_sim.py --P "$1" --M "$2" --N "$3" --L "$4"