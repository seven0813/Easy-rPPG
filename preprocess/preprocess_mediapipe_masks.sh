#!/usr/bin/env bash
set -euo pipefail

dataset_name="${1:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
preprocess_script="${script_dir}/script/preprocess_mediapipe_masks.py"
generate_h5_txt_script="${script_dir}/script/genetate_h5_txt.py"
num_workers="${NUM_WORKERS:-4}"
store_size=192
output_tag=m
crop_mode=frame        # crop_mode: frame or global
top_priority=0.85
bbox_margin=0.1

if [[ -z "${dataset_name}" ]]; then
    echo "Usage: bash preprocess_mediapipe_masks.sh <UBFC-rPPG|PURE|BUAA>"
    exit 1
fi

common_args=(
    --store_size "${store_size}"
    --bbox_margin "${bbox_margin}"
    --output_tag "${output_tag}"
    --num_workers "${num_workers}"
    --crop_mode "${crop_mode}"
    --top_priority "${top_priority}"
)

case "${dataset_name}" in
    UBFC-rPPG)
        # h5_dir="/path/to/h5/UBFC-rPPG/masks192"
        # h5_dir="/path/to/h5/UBFC-rPPG/mediapipe/m0t5"
        h5_dir="/path/to/h5/UBFC-rPPG/mediapipe/m1t85"
        # h5_dir="/path/to/h5/UBFC-rPPG/mediapipe/m1t85"
        python "${preprocess_script}" \
            --dataset_name "UBFC-rPPG" \
            --video_dir "/path/to/data/UBFC-rPPG" \
            --json_dir "/path/to/data/UBFC-rPPG" \
            --h5_dir "${h5_dir}" \
            "${common_args[@]}"
        ;;
    PURE)
        # h5_dir="/path/to/h5/PURE/masks192"
        # h5_dir="/path/to/h5/PURE/m192_mask"
        h5_dir="/path/to/h5/PURE/mediapipe/m1t85"
        python "${preprocess_script}" \
            --dataset_name "PURE" \
            --video_dir "/path/to/data/PURE" \
            --json_dir "/path/to/data/PURE" \
            --h5_dir "${h5_dir}" \
            "${common_args[@]}"
        ;;
    BUAA)
        h5_dir="/path/to/h5/BUAA/masks192"
        python "${preprocess_script}" \
            --dataset_name "BUAA" \
            --video_dir "/path/to/data/BUAA" \
            --json_dir "/path/to/data/BUAA" \
            --h5_dir "${h5_dir}" \
            "${common_args[@]}"
        ;;
    *)
        echo "Unsupported dataset_name: ${dataset_name}"
        echo "Supported: UBFC-rPPG, PURE, BUAA"
        exit 1
        ;;
esac

python "${generate_h5_txt_script}" \
    --root_dir "${h5_dir}" \
    --output_txt "${PWD}/${output_tag}_h5.txt" \
    --name_contains "_${output_tag}_s${store_size}.h5"



### m0t5: margins=0.0,top=50%
### m1t85: margins=0.10,top=85%
