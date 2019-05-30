#!/bin/bash
set -euo pipefail
min() {
    if [ $1 -lt $2 ]; then
        echo $1
    else
        echo $2
    fi
}

njob=400
cmd="srun -p cpu,cpu-kishin"


data=$1
logdir=${data}/logs
mkdir -p ${logdir}


. ./env.sh


njob=$(min ${njob} $(wc -l ${data}/wav_rir_noise.scp))
split="split -e --numeric-suffixes=1 -a$(echo -n ${njob} | wc -m) -n r/${njob}"

${split} ${data}/wav_rir_noise.scp ${logdir}/wav_rir_noise.scp.
ls ${logdir} > /dev/null

pids=() # initialize pids
for i in $(seq -f %0$(echo -n ${njob} | wc -m)g ${njob}); do
    ${cmd} python generate_ideal_target.py \
        --in-scp ${logdir}/wav_rir_noise.scp.${i} \
        --out-h5 ${logdir}/target.${i}.h5 \
        --out-info ${logdir}/info.${i}.txt &> ${logdir}/ideal.${i}.log &
    pids+=($!) # store background pids
done
i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
[ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

args=''
for i in $(seq -f %0$(echo -n ${njob} | wc -m)g ${njob}); do
    args+="${logdir}/target.${i}.h5 "
done
python merge_hdf5.py ${args} -o ${data}/target.h5
# rm -f ${logdir}/target.*.h5

for i in $(seq -f %0$(echo -n ${njob} | wc -m)g ${njob}); do
    cat ${logdir}/info.${i}.txt
done > ${data}/info.txt
