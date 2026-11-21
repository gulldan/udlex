#!/usr/bin/env bash
# The owner extended the GPU phase to 06:00 JST (21.11.2026): hold the 3.5-hour timeout
# until the bench ends or 06:00 comes, then let it go; gpu_phase.sh restarts vLLM after.
set -u
T=$1 B=$2 OUT=$3
END=$(TZ=Asia/Tokyo date -d "2026-11-21 06:00" +%s)
say() { echo "$(TZ=Asia/Tokyo date "+%F %T %Z") $*" | tee -a "$OUT/vllm.log"; }
kill -STOP "$T" && say "owner extended the vLLM pause to 06:00; 3.5-hour timeout held"
while kill -0 "$B" 2>/dev/null && [ "$(date +%s)" -lt "$END" ]; do sleep 20; done
kill -CONT "$T"
say "timeout released ($(kill -0 "$B" 2>/dev/null && echo "06:00 reached, bench stopped" || echo "bench finished"))"
