"""Run with the colab CLI's own Python. Print live Colab assignments and orphans (live, no local session).

  --unassign        release orphans, but keep A100 orphans: a timed-out assign leaves them, and the next try adopts
  --unassign --all  release every orphan (loop exit: nothing will adopt them)
exit 10 when an A100 orphan was kept (the caller should start the next try now, not sleep on a billed card).
"""
import datetime, json, sys
from pathlib import Path
LEDGER = Path(__file__).resolve().parents[1] / "outputs/colab/ledger.jsonl"
from colab_cli.client import Accelerator
from colab_cli.common import state
KEEP = {Accelerator.A100}
local = {s.endpoint for s in state.store.list().values()}
live = state.client.list_assignments()
orphans = [a for a in live if a.endpoint not in local]
kept = [a for a in orphans if a.accelerator in KEEP and "--all" not in sys.argv]
print(json.dumps({"live": len(live), "local": len(local), "orphans": [a.accelerator.value for a in orphans],
                  "kept": [a.accelerator.value for a in kept]}))
if "--unassign" in sys.argv:
    for a in orphans:
        if a in kept:
            continue
        state.client.unassign(a.endpoint)
        with open(str(LEDGER), "a") as f:
            f.write(json.dumps({"t": datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "event": "unassign_orphan", "accelerator": a.accelerator.value}) + "\n")
        print("unassigned orphan", a.accelerator.value)
sys.exit(10 if kept else 0)
