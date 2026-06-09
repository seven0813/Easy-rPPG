#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-}"
if [[ -z "${MODE}" ]]; then
  echo "Usage: $0 <video|pure>" >&2
  exit 1
fi
shift || true

OPENFACE_BIN=""
#TODO: 需要先安装OPENFACE，并将OPENFACE_BIN修改为实际的路径
#TODO: 下面的路径要改为实际的路径

case "${MODE}" in
  video)
    ## UBFC-rPPG
    INPUT_ROOT=""
    OUTPUT_ROOT=""
    PATTERN='vid.avi'

    python script/extract_openface_landmarks.py \
      --mode video \
      --openface_bin "${OPENFACE_BIN}" \
      --input_root "${INPUT_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      --pattern "${PATTERN}" \
      --recursive \
      --two_d_only \
      "$@"
    ;;
  pure)
    ## PURE
    INPUT_ROOT=""
    OUTPUT_ROOT=""
    
    python script/extract_openface_landmarks.py \
      --mode pure \
      --openface_bin "${OPENFACE_BIN}" \
      --input_root "${INPUT_ROOT}" \
      --output_root "${OUTPUT_ROOT}" \
      --two_d_only \
      # --skip_existing \
      "$@"
    ;;
  *)
    echo "Unknown mode: ${MODE}. Use video or pure." >&2
    exit 1
    ;;
esac
