#!/usr/bin/env bash
# The owner lifted the GPU-phase limit (21.11.2026): hold the timeout until the bench ends.
set -u
T=$1 B=$2 OUT=$3
say() { echo "$(TZ=Asia/Tokyo date "+%F %T %Z") $*" | tee -a "$OUT/vllm.log"; }
kill -STOP "$T"; say "owner lifted the limit: timeout held until the GPU bench ends"
while kill -0 "$B" 2>/dev/null; do sleep 20; done
kill -CONT "$T"; say "GPU bench finished; timeout released"
