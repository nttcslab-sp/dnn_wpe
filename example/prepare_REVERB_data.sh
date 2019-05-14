#!/bin/bash
set -euo pipefail

stage=3

WSJ_CAM0=/data/rigel1/corpora/REVERB_DATA_OFFICIAL/wsjcam0
REVERB_DATA_OFFICIAL=/data/rigel1/corpora/REVERB_DATA_OFFICIAL
dir=data


if [ "${stage}" -le 1 ]; then
    echo "$0: Stage1: Convert sph to wav format in WSJ_CAM0 training data"
    mkdir -p ${dir}/downloads

    if ! which sph2pipe &> /dev/null; then
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5.tar.gz ]; then
            wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz -O ${dir}/downloads/sph2pipe_v2.5.tar.gz
        fi
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5/sph2pipe ]; then
            tar zxvf ${dir}/downloads/sph2pipe_v2.5.tar.gz -C ${dir}/downloads
            pushd ${dir}/downloads/sph2pipe_v2.5; gcc -o sph2pipe *.c -lm; pushd
        fi
        export PATH=$(cd ${dir}; pwd)/downloads/sph2pipe_v2.5:${PATH}
    fi

    abs_dir=$(cd ${dir}; pwd)
    pushd ${WSJ_CAM0}
    for wav in $(ls data/primary_microphone/si_tr/*/*.wv1); do
        outwav=${wav/.wv1/.wav}
        mkdir -p $(dirname ${abs_dir}/wsjcam0/${outwav})
        sph2pipe -f wav ${wav} ${abs_dir}/wsjcam0/${outwav}
    done
    pushd
fi


if [ "${stage}" -le 2 ]; then
    echo "$0: Stage2: Prepare scp files"

    mkdir -p ${dir}/SimData_tr
    for reverb_wav in $(ls ${REVERB_DATA_OFFICIAL}/REVERB_WSJCAM0_tr/data/mc_train/primary_microphone/si_tr/*/*_ch1.wav); do
        uttid=$(echo ${reverb_wav} | sed -e 's#^.*/##' | sed -e 's/_ch1.wav//')
        clean_wav=${dir}/wsjcam0/data/primary_microphone/si_tr/$(echo ${reverb_wav} | sed -e 's/_ch1.wav/.wav/' | awk -F'/' '{ print $(NF-1) "/" $NF }')

        echo "${uttid} ${reverb_wav} ${reverb_wav/_ch1.wav/_ch2.wav} ${reverb_wav/_ch1.wav/_ch3.wav} ${reverb_wav/_ch1.wav/_ch4.wav} ${reverb_wav/_ch1.wav/_ch5.wav} ${reverb_wav/_ch1.wav/_ch6.wav} ${reverb_wav/_ch1.wav/_ch7.wav} ${reverb_wav/_ch1.wav/_ch8.wav}" >&3
        echo "${uttid} ${clean_wav}" >&4
    done 3> ${dir}/SimData_tr/reverb_mono.scp 4> ${dir}/SimData_tr/clean.scp

    for taskfile in RealData_dt_for_8ch_far_room1 RealData_et_for_8ch_near_room1 SimData_dt_for_8ch_far_room3 SimData_dt_for_8ch_near_room3 SimData_et_for_8ch_far_room3 SimData_et_for_8ch_near_room3 RealData_dt_for_8ch_near_room1 SimData_dt_for_8ch_far_room1 SimData_dt_for_8ch_near_room1 SimData_et_for_8ch_far_room1 SimData_et_for_8ch_near_room1 RealData_et_for_8ch_far_room1 SimData_dt_for_8ch_far_room2 SimData_dt_for_8ch_near_room2 SimData_et_for_8ch_far_room2 SimData_et_for_8ch_near_room2; do
        mkdir -p ${dir}/${taskfile}
        for ch in A B C D E F G H; do
            for base in $(cat ${REVERB_DATA_OFFICIAL}/taskFiles/8ch/${taskfile}_${ch}); do

                if echo ${taskfile} | grep RealData_dt &> /dev/null; then
                    echo "${REVERB_DATA_OFFICIAL}/MC_WSJ_AV_Dev${base}"

                elif echo ${taskfile} | grep RealData_et &> /dev/null; then
                    echo "${REVERB_DATA_OFFICIAL}/MC_WSJ_AV_Eval${base}"

                elif echo ${taskfile} | grep SimData_dt &> /dev/null; then
                    echo "${REVERB_DATA_OFFICIAL}/REVERB_WSJCAM0_dt/data${base}"

                elif echo ${taskfile} | grep SimData_et &> /dev/null; then
                    echo "${REVERB_DATA_OFFICIAL}/REVERB_WSJCAM0_et/data${base}"
                fi

            done > ${dir}/${taskfile}/${ch}.list
        done

        if echo ${taskfile} | grep SimData_dt &> /dev/null; then
            for base in $(cat ${REVERB_DATA_OFFICIAL}/taskFiles/8ch/$(echo ${taskfile} | sed -e 's/for_8ch_[a-z]*_/for_cln_/')); do
                uttid=$(echo ${base} | sed -e 's#^.*/##' | sed -e 's/.wav//')
                echo "${uttid} ${REVERB_DATA_OFFICIAL}/REVERB_WSJCAM0_dt/data${base}"
            done

        elif echo ${taskfile} | grep SimData_et &> /dev/null; then
            for base in $(cat ${REVERB_DATA_OFFICIAL}/taskFiles/8ch/$(echo ${taskfile} | sed -e 's/for_8ch_[a-z]*_/for_cln_/')); do
                uttid=$(echo ${base} | sed -e 's#^.*/##' | sed -e 's/.wav//')
                echo "${uttid} ${REVERB_DATA_OFFICIAL}/REVERB_WSJCAM0_et/data${base}"
            done
        fi > ${dir}/${taskfile}/clean.scp

        # Create uttid.list
        # e.g. /audio/stat/T7/array2/5k/AMI_WSJ17-Array2-4_T7c0210_ch1.wav -> t7c0210
        <${dir}/${taskfile}/A.list sed -e 's#^.*/##' | sed -e 's/.wav$//' | sed -e 's/_ch[0-9]$//' | sed -e 's/^.*_//' | tr '[A-Z]' '[a-z]' > ${dir}/${taskfile}/uttid.list

        paste ${dir}/${taskfile}/uttid.list ${dir}/${taskfile}/A.list ${dir}/${taskfile}/B.list ${dir}/${taskfile}/C.list ${dir}/${taskfile}/D.list ${dir}/${taskfile}/E.list ${dir}/${taskfile}/F.list ${dir}/${taskfile}/G.list ${dir}/${taskfile}/H.list | sort -k1 > ${dir}/${taskfile}/reverb_mono.scp
    done
fi


if [ "${stage}" -le 3 ]; then
    echo "$0: Stage3: Mix mono channel wav files into a multi channel wav files"

    if ! which sox &> /dev/null; then
        echo "Please install sox."
        exit 1
    fi

    for taskfile in SimData_tr RealData_dt_for_8ch_far_room1 RealData_et_for_8ch_near_room1 SimData_dt_for_8ch_far_room3 SimData_dt_for_8ch_near_room3 SimData_et_for_8ch_far_room3 SimData_et_for_8ch_near_room3 RealData_dt_for_8ch_near_room1 SimData_dt_for_8ch_far_room1 SimData_dt_for_8ch_near_room1 SimData_et_for_8ch_far_room1 SimData_et_for_8ch_near_room1 RealData_et_for_8ch_far_room1 SimData_dt_for_8ch_far_room2 SimData_dt_for_8ch_near_room2 SimData_et_for_8ch_far_room2 SimData_et_for_8ch_near_room2; do
        while read line <&3; do
            uttid=$(echo ${line} | awk '{ print $1 }' )
            outwav=${dir}/${taskfile}/wavs/$(echo ${line} | awk -F'/' '{ print $(NF-1) "/" $NF }' | sed -e 's/_ch[0-9].wav$/.wav/')
            mkdir -p $(dirname ${outwav})
            sox -M $(echo ${line} | awk '{ for(i=2;i<=NF;++i){ print $i " " } }') ${outwav}

            echo "${uttid} ${outwav}" >&4
        done 3<${dir}/${taskfile}/reverb_mono.scp 4> ${dir}/${taskfile}/reverb.scp
    done
fi


if [ "${stage}" -le 4 ]; then
    echo "$0: Stage4: Split SimData_tr into tr and dev sets"
    mkdir -p ${dir}/train ${dir}/dev
    for t in reverb clean; do
        head -n -400 ${dir}/SimData_tr/${t}.scp > ${dir}/train/${t}.scp
        tail -n 400 ${dir}/SimData_tr/${t}.scp > ${dir}/dev/${t}.scp
    done
fi
