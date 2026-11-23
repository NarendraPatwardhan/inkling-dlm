#!/bin/bash
# Try for a Colab A100 (40 GB is fine; standard then high-mem) every 5 minutes, no cutoff.
# Orphans (assigned server-side, no local session) are released, except A100s: a timed-out assign leaves those and
# the next try adopts them (scripts/colab_assign.py), so the wait is cut short the moment one shows up.
# Once a card is held, colab_ctl runs the gates within the CU budget and keeps the session. Never trains.
# usage: scripts/a100_retry.sh <commit>   (run detached: tmux new -d -s inkling-gates ...)
cd "$(dirname "$0")/.."
SHA=${1:?commit}
LOG=outputs/colab/retry; mkdir -p $LOG
CPY=${COLAB_PY:-$HOME/.local/share/uv/tools/google-colab-cli/bin/python}
orph() { $CPY scripts/colab_orphans.py --unassign "$@" 2>&1 | tail -3; return ${PIPESTATUS[0]}; }
i=$(ls $LOG/attempt-*.log 2>/dev/null | wc -l)
trap 'echo "=== stopping $(date -Is) ==="; orph --all; echo "RETRY_EXIT rc=signal attempt=$i"; exit 1' INT TERM HUP
while true; do
  i=$((i+1)); t0=$(date +%s)
  echo "=== attempt $i $(date -Is) ==="
  orph
  .venv/bin/python scripts/colab_ctl.py gates --commit "$SHA" --cu-budget 5 > $LOG/attempt-$i.log 2>&1
  rc=$?
  echo "attempt $i rc=$rc $(grep -ho '"how": "[a-z]*"' $LOG/attempt-$i.log | head -1)"
  grep -q "no A100 available" $LOG/attempt-$i.log || break
  while [ $((300 - ($(date +%s) - t0))) -gt 0 ]; do      # wait out the 5 minutes, watching for a late A100
    sleep 30; orph; [ $? -eq 10 ] && { echo "late A100 orphan: adopting now"; break; }
  done
done
sleep 30
echo "=== final check $(date -Is) ==="
orph --all
colab sessions 2>&1 | grep -q "\[inkling-a100\]" && echo "A100 session inkling-a100 kept"
colab usage 2>&1 | grep -E "balance|rate|assignments"
echo "RETRY_EXIT rc=$rc attempt=$i"
