#!/bin/bash -l
# Shared generator core: make N files of size S totaling 10GB, sharded into
# subdirs of 1000 files each. Sourced/called by the PBS wrappers.
# Usage: gen_one <file_size> <dest_dir>
#   file_size: dd-style (4K, 64K, 256K, 1M, 4M, 10M)
# Total is fixed at 10GB; file count = 10GB / file_size.

set -eu

SHARD_SIZE=1000
TOTAL_BYTES=$((10 * 1024 * 1024 * 1024))   # 10 GiB

bytes_of() {
  case "$1" in
    *G) echo $(( ${1%G} * 1024 * 1024 * 1024 ));;
    *M) echo $(( ${1%M} * 1024 * 1024 ));;
    *K) echo $(( ${1%K} * 1024 ));;
    *)  echo "$1";;
  esac
}

gen_one() {
  local SIZE="$1" DEST="$2"
  local PER COUNT
  PER=$(bytes_of "${SIZE}")
  COUNT=$(( TOTAL_BYTES / PER ))

  echo "[$(date '+%H:%M:%S')] ${SIZE} x ${COUNT} files -> ${DEST} (10GB total, sharded ${SHARD_SIZE}/dir)"
  mkdir -p "${DEST}"

  # resumable: if the expected count already exists, skip
  local have
  have=$(find "${DEST}" -type f -name 'f_*.bin' 2>/dev/null | wc -l)
  if [ "${have}" -ge "${COUNT}" ]; then
    echo "[$(date '+%H:%M:%S')]   already have ${have} files, skipping"
    return 0
  fi

  local i shard shard_dir fname
  i=0
  while [ "${i}" -lt "${COUNT}" ]; do
    shard=$(( i / SHARD_SIZE ))
    shard_dir=$(printf "%s/shard_%05d" "${DEST}" "${shard}")
    mkdir -p "${shard_dir}"
    fname=$(printf "%s/f_%08d.bin" "${shard_dir}" "${i}")
    if [ ! -f "${fname}" ]; then
      head -c "${PER}" /dev/urandom > "${fname}"
    fi
    i=$(( i + 1 ))
    if [ $(( i % 100000 )) -eq 0 ]; then
      echo "[$(date '+%H:%M:%S')]   ${i}/${COUNT} ..."
    fi
  done

  local final
  final=$(find "${DEST}" -type f -name 'f_*.bin' | wc -l)
  echo "[$(date '+%H:%M:%S')]   done: ${final} files in ${DEST}"
}
