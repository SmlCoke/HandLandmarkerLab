#!/usr/bin/env bash
set -Eeuo pipefail

# Required per-screen variables.
: "${HLMF_MEMBER:?Set HLMF_MEMBER, for example: peak}"
: "${HLMF_SOURCE_NAME:?Set HLMF_SOURCE_NAME, for example: fist_side_r01}"

# Optional variables.
HLMF_NEGATIVE_CANDIDATE_THRESHOLD="${HLMF_NEGATIVE_CANDIDATE_THRESHOLD:-}"
HLMF_ROOT="${HLMF_ROOT:-/root/HandLandmarksFab}"
HAND_DATASET_ROOT="${HAND_DATASET_ROOT:-/root/autodl-tmp/DatesetFab}"
HLMF_LOG_ROOT="${HLMF_LOG_ROOT:-/root/hlmf_autolabel_logs}"

if [[ ! "${HLMF_MEMBER}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: HLMF_MEMBER contains invalid characters" >&2
  exit 1
fi
if [[ ! "${HLMF_SOURCE_NAME}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "ERROR: HLMF_SOURCE_NAME contains invalid characters" >&2
  exit 1
fi

HLMF_SOURCE_ROOT="${HAND_DATASET_ROOT}/PretrainSource/HandViolencePro0719/${HLMF_MEMBER}/${HLMF_SOURCE_NAME}"
IMAGES_DIR="${HLMF_SOURCE_ROOT}/images"

mkdir -p "${HLMF_LOG_ROOT}"
LOG_FILE="${HLMF_LOG_ROOT}/${HLMF_MEMBER}__${HLMF_SOURCE_NAME}.log"
exec > >(tee -a "${LOG_FILE}") 2>&1

echo "===== HLMF AutoLabel: $(date '+%F %T') ====="
echo "member:      ${HLMF_MEMBER}"
echo "source:      ${HLMF_SOURCE_NAME}"
echo "source root: ${HLMF_SOURCE_ROOT}"
echo "log:         ${LOG_FILE}"

if [[ ! -d "${HLMF_ROOT}" ]]; then
  echo "ERROR: HLMF repository not found: ${HLMF_ROOT}" >&2
  exit 1
fi
if [[ ! -d "${IMAGES_DIR}" ]]; then
  echo "ERROR: input images directory not found: ${IMAGES_DIR}" >&2
  exit 1
fi

TIFF_COUNT="$(
  find "${IMAGES_DIR}" -maxdepth 1 -type f \
    \( -iname '*.tif' -o -iname '*.tiff' \) | wc -l
)"
if [[ "${TIFF_COUNT}" -eq 0 ]]; then
  echo "ERROR: no TIFF images found in ${IMAGES_DIR}" >&2
  exit 1
fi
echo "input TIFF:  ${TIFF_COUNT}"

# Prevent two screens from processing the same source on one server.
LOCK_FILE="/tmp/hlmf_${HLMF_MEMBER}_${HLMF_SOURCE_NAME}.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "ERROR: this source is already running; lock: ${LOCK_FILE}" >&2
  exit 1
fi

source /root/miniconda3/etc/profile.d/conda.sh
conda activate anfab
cd "${HLMF_ROOT}"

export HAND_DATASET_ROOT HLMF_SOURCE_ROOT

if [[ -n "${HLMF_NEGATIVE_CANDIDATE_THRESHOLD}" ]]; then
  if [[ ! "${HLMF_NEGATIVE_CANDIDATE_THRESHOLD}" =~ ^0(\.[0-9]+)?$|^1(\.0+)?$ ]]; then
    echo "ERROR: HLMF_NEGATIVE_CANDIDATE_THRESHOLD must be within [0,1]" >&2
    exit 1
  fi
  AUTOLABEL_OVERRIDES="$(printf \
    '{"palm":{"negative_candidate_threshold":%s}}' \
    "${HLMF_NEGATIVE_CANDIDATE_THRESHOLD}")"
else
  AUTOLABEL_OVERRIDES='{}'
fi

echo "overrides:   ${AUTOLABEL_OVERRIDES}"

make autolabel \
  HLMF_SOURCE_ROOT="${HLMF_SOURCE_ROOT}" \
  AUTOLABEL_ROLE=train \
  AUTOLABEL_OVERRIDES="${AUTOLABEL_OVERRIDES}"

LABELS="${HLMF_SOURCE_ROOT}/02_roi_crops/hand_landmarks_autolabel_draft.jsonl"
MANIFEST="${HLMF_SOURCE_ROOT}/02_roi_crops/hand_roi_crops_manifest.jsonl"
ROI_DIR="${HLMF_SOURCE_ROOT}/02_roi_crops/images"
test -f "${LABELS}"
test -f "${MANIFEST}"
test -d "${ROI_DIR}"

export HLMF_RESULT_LABELS="${LABELS}"
python - <<'PY'
import json
import os
from pathlib import Path

labels = Path(os.environ["HLMF_RESULT_LABELS"])
counts = {True: 0, False: 0, None: 0}
with labels.open("r", encoding="utf-8") as handle:
    for line in handle:
        if not line.strip():
            continue
        row = json.loads(line)
        present = (row.get("hand_presence") or {}).get("present")
        counts[present if present in (True, False) else None] += 1

print("MediaPipe positive:", counts[True])
print("MediaPipe abstain/negative:", counts[False])
print("Invalid presence:", counts[None])
PY

ROI_COUNT="$(find "${ROI_DIR}" -maxdepth 1 -type f | wc -l)"
echo "generated ROI: ${ROI_COUNT}"
echo "result root:   ${HLMF_SOURCE_ROOT}"
echo "===== AutoLabel complete: $(date '+%F %T') ====="
