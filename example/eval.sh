#!/bin/bash
set -euo pipefail

stage=5
stop_stage=10000000
njob=30
ngpu=1
cmd="srun -p gpu --grep gpu:${ngpu}"


data=$1
model_file=$2
dir=$3
logdir=${dir}/logs


. ./env.sh


split="split -e --numeric-suffixes=1 -a$(echo -n ${njob} | wc -m) -n r/${njob}"

if [ "${stage}" -le 1 ] && [ "${stop_stage}" -ge 1 ]; then
    echo "$0: Stage1: apply WPE"

    ${split} ${data}/reverb.scp ${logdir}/reverb.scp.

    pids=() # initialize pids
    for i in $(seq ${njob}); do
        if [ -e ${logdir}/reverb.scp.${i} ]; then
            ${cmd} python decode.py \
                --in-scp ${logdir}/reverb.scp.${i} \
                --out-dir ${logdir}/enhanced.${i} \
                --model-file ${model_file} \
                --ngpu ${ngpu} > ${logdir}/decode.${i}.log &
            pids+=($!) # store background pids
        fi
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for i in $(seq ${njob}); do
        if [ -e ${logdir}/reverb.scp.${i} ]; then
            cat ${logdir}/enhanced.${i}/wav.scp
        fi
    done > ${dir}/enhanced.scp

fi


if [ "${stage}" -le 2 ] && [ "${stop_stage}" -ge 2 ]; then
    echo "$0: Stage2: evaluate"

    ${split} ${dir}/enhanced.scp ${logdir}/enhanced.scp.

    pids=() # initialize pids
    for i in $(seq ${njob}); do
        if [ -e ${logdir}/enhanced.scp.${i} ]; then
            ${cmd} python decode.py \
                --in-scp ${logdir}/enhanced.scp.${i}
            pids+=($!) # store background pids
        fi
    done
    i=0; for pid in "${pids[@]}"; do wait ${pid} || ((i++)); done
    [ ${i} -gt 0 ] && echo "$0: ${i} background jobs are failed." && false

    for i in $(seq ${njob}); do
        if [ -e ${logdir}/enhanced.scp.${i} ]; then
            cat ${logdir}/some.${i}/
        fi
    done > ${dir}/some
fi
