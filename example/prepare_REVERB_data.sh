#!/bin/bash
set -euo pipefail
if [ $# -ne 0 ]; then
    echo "$0: Error: Not supporting arguments"
    exit 1
fi
if ! which sox &> /dev/null; then
    echo "Please install sox."
    exit 1
fi

stage=0
stop_stage=6

WSJ_CAM0=/data/rigel1/corpora/REVERB_DATA_OFFICIAL/wsjcam0
REVERB_DATA_OFFICIAL=/data/rigel1/corpora/REVERB_DATA_OFFICIAL
dir=data

if [ "${stage}" -le 0 ] && [ ${stop_stage} -ge 0 ]; then
    echo "$0: Stage0: Install PESQ"

    wget -O data/downloads/P862.zip https://www.itu.int/rec/dologin_pub.asp\?lang\=e\&id\=T-REC-P.862-200102-I\!\!SOFT-ZST-E\&type\=items
    unzip data/downloads/P862.zip -d data/downloads
    ( cd data/downloads/P862/Software/source; gcc -o PESQ *.c -lm; )
fi

if [ "${stage}" -le 1 ] && [ ${stop_stage} -ge 1 ]; then
    echo "$0: Stage1: Convert sph to wav format in WSJ_CAM0 training data"
    mkdir -p ${dir}/downloads

    if ! which sph2pipe &> /dev/null; then
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5.tar.gz ]; then
            wget https://www.ldc.upenn.edu/sites/www.ldc.upenn.edu/files/ctools/sph2pipe_v2.5.tar.gz -O ${dir}/downloads/sph2pipe_v2.5.tar.gz
        fi
        if [ ! -e ${dir}/downloads/sph2pipe_v2.5/sph2pipe ]; then
            tar zxvf ${dir}/downloads/sph2pipe_v2.5.tar.gz -C ${dir}/downloads
            ( cd ${dir}/downloads/sph2pipe_v2.5; gcc -o sph2pipe *.c -lm; )
        fi
        export PATH=$(cd ${dir}; pwd)/downloads/sph2pipe_v2.5:${PATH}
    fi

    abs_dir=$(cd ${dir}; pwd)
    (
    cd ${WSJ_CAM0}
    for wav in $(ls data/primary_microphone/si_tr/*/*.wv1); do
        outwav=${wav/.wv1/.wav}
        mkdir -p $(dirname ${abs_dir}/wsjcam0/${outwav})
        sph2pipe -f wav ${wav} ${abs_dir}/wsjcam0/${outwav}
    done
    )

    mkdir -p ${dir}/material
    for wav in $(ls ${dir}/wsjcam0/data/primary_microphone/si_tr/*/*.wav); do
        echo "$(basename ${wav%.*}) ${wav}"
    done > ${dir}/material/wav.scp

fi


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
    mkdir -p ${dir}/downloads

    for f in reverb_tools_for_Generate_mcTrainData; do
        if [ ! -e ${dir}/downloads/${f} ]; then
            if [ ! -e ${dir}/downloads/${f}.tgz ]; then
                wget https://reverb2014.dereverberation.com/tools/${f}.tgz -O ${dir}/downloads/${f}.tgz
            fi

            tar zxvf ${dir}/downloads/${f}.tgz -C ${dir}/downloads \
                || { rm -f ${dir}/downloads/${f}.tgz;
                     wget https://reverb2014.dereverberation.com/tools/${f}.tgz -O ${dir}/downloads/${f}.tgz;
                     tar zxvf ${dir}/downloads/${f}.tgz -C ${dir}/downloads; }
        fi
    done
fi


if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
    mkdir -p ${dir}/material
    base=${dir}/downloads/reverb_tools_for_Generate_mcTrainData/RIR
    for d in SmallRoom1 SmallRoom2 MediumRoom1 MediumRoom2 LargeRoom1 LargeRoom2; do
        cat << EOF
${d} ${base}/RIR_${d}_near_AnglA.wav ${base}/RIR_${d}_far_AnglA.wav ${base}/RIR_${d}_near_AnglB.wav ${base}/RIR_${d}_far_AnglB.wav
EOF
    done > ${dir}/material/rir.list

    for d in SmallRoom1 SmallRoom2 MediumRoom1 MediumRoom2 LargeRoom1 LargeRoom2; do
        cat << EOF
${d}$(for i in {1..10}; do echo -n " ${dir}/downloads/reverb_tools_for_Generate_mcTrainData/NOISE/Noise_${d}_${i}.wav"; done)
EOF
    done > ${dir}/material/noise.list

    mkdir -p ${dir}/train ${dir}/dev
    head -n -400 ${dir}/material/wav.scp > ${dir}/train/wav.scp
    tail -n 400 ${dir}/material/wav.scp > ${dir}/dev/wav.scp

    python << EOF
#!/usr/bin/env pythoh
from dataset import paring_wav_rir_noise
paring_wav_rir_noise('${dir}/train/wav.scp',
                     '${dir}/material/rir.list',
                     '${dir}/material/noise.list',
                     '${dir}/train/wav_rir_noise.scp')
paring_wav_rir_noise('${dir}/dev/wav.scp',
                     '${dir}/material/rir.list',
                     '${dir}/material/noise.list',
                     '${dir}/dev/wav_rir_noise.scp')
EOF

fi


if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
    echo "$0: Stage4: Create utt2nframes"
    for d in ${dir}/train ${dir}/dev; do
        while read line <&3; do
            uttid=$(echo ${line} | cut -d" " -f1)
            wav=$(echo ${line} | cut -d" " -f2-)
            nframes=$(soxi -s ${wav})
            echo "${uttid} ${nframes}"
        done 3<${d}/wav.scp >${d}/utt2nframes
    done
fi



if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
    echo "$0: Stage5: Prepare scp files"

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


if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
    echo "$0: Stage6: Mix mono channel wav files into a multi channel wav files"

    for taskfile in RealData_dt_for_8ch_far_room1 RealData_et_for_8ch_near_room1 SimData_dt_for_8ch_far_room3 SimData_dt_for_8ch_near_room3 SimData_et_for_8ch_far_room3 SimData_et_for_8ch_near_room3 RealData_dt_for_8ch_near_room1 SimData_dt_for_8ch_far_room1 SimData_dt_for_8ch_near_room1 SimData_et_for_8ch_far_room1 SimData_et_for_8ch_near_room1 RealData_et_for_8ch_far_room1 SimData_dt_for_8ch_far_room2 SimData_dt_for_8ch_near_room2 SimData_et_for_8ch_far_room2 SimData_et_for_8ch_near_room2; do
        while read line <&3; do
            uttid=$(echo ${line} | awk '{ print $1 }' )
            outwav=${dir}/${taskfile}/wavs/$(echo ${line} | awk -F'/' '{ print $(NF-1) "/" $NF }' | sed -e 's/_ch[0-9].wav$/.wav/')
            mkdir -p $(dirname ${outwav})
            sox -M $(echo ${line} | awk '{ for(i=2;i<=NF;++i){ print $i " " } }') ${outwav}

            echo "${uttid} ${outwav}" >&4
        done 3<${dir}/${taskfile}/reverb_mono.scp 4> ${dir}/${taskfile}/reverb.scp
    done
fi
