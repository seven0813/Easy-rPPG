#!/usr/bin/env bash
set -euo pipefail

dataset_name="${1:-}"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
preprocess_script="${script_dir}/script/preprocess_toolbox.py"
generate_h5_txt_script="${script_dir}/script/genetate_h5_txt.py"

# rPPG-Toolbox CROP_FACE defaults. Edit here or invoke the Python entry directly.
do_crop_face=true
backend=HC                    # HC or Y5F
use_large_face_box=true
large_box_coef=1.5
do_dynamic_detection=false
dynamic_detection_frequency=32
use_median_face_box=false     # effective only with dynamic detection

# Easy-rPPG output/runtime settings.
store_size=128
output_tag=toolbox
preview_frames=2
num_workers="${NUM_WORKERS:-1}"
device="${DEVICE:-}"          # empty = automatic; examples: cpu, cuda:0

if [[ -z "${dataset_name}" ]]; then
    echo "Usage: bash preprocess/preprocess_toolbox.sh <UBFC-rPPG|PURE>"
    exit 1
fi

common_args=(
    --backend "${backend}"
    --large_box_coef "${large_box_coef}"
    --dynamic_detection_frequency "${dynamic_detection_frequency}"
    --store_size "${store_size}"
    --output_tag "${output_tag}"
    --preview_frames "${preview_frames}"
    --num_workers "${num_workers}"
)

if [[ "${do_crop_face}" == true ]]; then
    common_args+=(--do_crop_face)
else
    common_args+=(--no-do_crop_face)
fi
if [[ "${use_large_face_box}" == true ]]; then
    common_args+=(--use_large_face_box)
else
    common_args+=(--no-use_large_face_box)
fi
if [[ "${do_dynamic_detection}" == true ]]; then
    common_args+=(--do_dynamic_detection)
else
    common_args+=(--no-do_dynamic_detection)
fi
if [[ "${use_median_face_box}" == true ]]; then
    common_args+=(--use_median_face_box)
else
    common_args+=(--no-use_median_face_box)
fi
if [[ -n "${device}" ]]; then
    common_args+=(--device "${device}")
fi

case "${dataset_name}" in
    UBFC-rPPG)
        h5_dir="/path/to/h5/UBFC-rPPG/toolbox"
        python "${preprocess_script}" \
            --dataset_name "UBFC-rPPG" \
            --video_dir "/path/to/data/UBFC-rPPG" \
            --json_dir "/path/to/data/UBFC-rPPG" \
            --h5_dir "${h5_dir}" \
            "${common_args[@]}"
        ;;
    PURE)
        h5_dir="/path/to/h5/PURE/toolbox"
        python "${preprocess_script}" \
            --dataset_name "PURE" \
            --video_dir "/path/to/data/PURE" \
            --json_dir "/path/to/data/PURE" \
            --h5_dir "${h5_dir}" \
            "${common_args[@]}"
        ;;
    *)
        echo "Unsupported dataset_name: ${dataset_name}"
        echo "Supported: UBFC-rPPG, PURE"
        exit 1
        ;;
esac

python "${generate_h5_txt_script}" \
    --root_dir "${h5_dir}" \
    --output_txt "${PWD}/${output_tag}_${dataset_name}_h5.txt" \
    --name_contains "_${output_tag}_s${store_size}.h5"
