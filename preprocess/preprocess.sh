#!/usr/bin/env bash
set -e

dataset_name="$1"

if [ -z "$dataset_name" ]; then
    echo "Usage: bash preprocess.sh <UBFC-rPPG|PURE|BUAA>"
    exit 1
fi

if [ "$dataset_name" = "UBFC-rPPG" ]; then
    # UBFC-rPPG
    python script/preprocess.py \
    #TODO: 这里的路径需要修改为实际的路径
        --dataset_name "UBFC-rPPG" \
        --video_dir "" \
        --json_dir "" \
        --landmark_dir "" \
        --h5_dir "" \
        --store_size 128

elif [ "$dataset_name" = "PURE" ]; then
    # PURE
    python script/preprocess.py \
    #TODO: 这里的路径需要修改为实际的路径
        --dataset_name "PURE" \
        --video_dir "" \
        --landmark_dir "" \
        --json_dir "" \
        --h5_dir "" \
        --store_size 128

elif [ "$dataset_name" = "BUAA" ]; then
    # # BUAA
    python script/preprocess.py \
        --dataset_name "BUAA" \
        --video_dir "" \
        --landmark_dir "" \
        --json_dir "" \
        --h5_dir "" \
        --store_size 128

else
    echo "Unsupported dataset_name: $dataset_name"
    echo "Supported: UBFC-rPPG, PURE, BUAA"
    exit 1
fi
