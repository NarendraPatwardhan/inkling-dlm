#!/bin/bash
# Try for a Colab A100 (40 GB is fine; standard then high-mem) every 5 minutes, no cutoff.
# Before each try: release orphan assignments and any stale inkling-a100 session.
# Once a card is assigned, colab_ctl runs the gates within the CU budget, then stops the session. Never trains.
# usage: scripts/a100_retry.sh <commit>   (run detached: tmux new -d -s inkling-gates ...)
cd "$(dirname "$0")/.."
SHA=${1:?commit}
LOG=outputs/colab/retry; mkdir -p $LOG
CPY=${COLAB_PY:-$HOME/.local/share/uv/tools/google-colab-cli/bin/python}
ORPH="$CPY scripts/colab_orphans.py --unassign"
i=$(ls $LOG/attempt-*.log 2>/dev/null | wc -l)
while true; do
  i=$((i+1)); t0=$(date +%s)
  echo "=== attempt $i $(date -Is) ==="
  $ORPH 2>&1 | tail -3
  if colab sessions 2>&1 | grep -q "\[inkling-a100\]"; then
    echo "stale inkling-a100 session: stopping it first"; colab stop -s inkling-a100 >/dev/null 2>&1
  fi
  .venv/bin/python scripts/colab_ctl.py gates --commit "$SHA" --cu-budget 5 > $LOG/attempt-$i.log 2>&1
  rc=$?
  echo "attempt $i rc=$rc"
  grep -q "no A100 available" $LOG/attempt-$i.log || break
  sleep 30; $ORPH 2>&1 | tail -3
  left=$((300 - ($(date +%s) - t0))); [ $left -gt 0 ] && sleep $left
done
sleep 30
echo "=== final check $(date -Is) ==="
$ORPH 2>&1 | tail -3
colab usage 2>&1 | grep -E "balance|rate|assignments"
echo "RETRY_EXIT rc=$rc attempt=$i"
