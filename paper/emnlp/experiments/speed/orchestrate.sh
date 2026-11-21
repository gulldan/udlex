#!/usr/bin/env bash
# After the GPU phase: keep vLLM off (the owner allowed it for as long as the measurement
# needs, 21.11.2026), time udlex on the hybrid release, then start vLLM again.
set -u
S=/mnt/hdd-4tb/codex-runs/udlex-speed-20261119
OUT=$1
VLLM=${UDLEX_SPEED_PAUSE:-}  # the vLLM container
say() { echo "$(TZ=Asia/Tokyo date "+%F %T %Z") $*" | tee -a "$OUT/vllm.log"; }
while tmux has-session -t speed-all-gpu 2>/dev/null; do sleep 20; done
say "GPU phase over; vLLM: $(docker ps --filter "name=^/$VLLM\$" --format "{{.Status}}")"
docker stop "$VLLM" >/dev/null 2>&1
say "vLLM stopped for the udlex pass: $(docker inspect -f "{{.State.Status}}" "$VLLM")"
sleep 60
/mnt/hdd-4tb/codex-runs/udlex-release-hybrid-20261120/udlex_c9.sh 2>&1 | tee -a /mnt/hdd-4tb/codex-runs/udlex-release-hybrid-20261120/udlex_c9.log
"$S/.venv-udlex/bin/python" -I "$S/harness/speed/bench.py" summary "$OUT" > "$OUT/summary.log" 2>&1 || say "no summary"
docker start "$VLLM" >/dev/null 2>&1; sleep 15
say "all C9 timing done; vLLM started: $(docker ps --filter "name=^/$VLLM\$" --format "{{.Status}}")"
