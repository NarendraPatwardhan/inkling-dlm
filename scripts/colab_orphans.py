"""Run with the colab CLI's own Python. Print live Colab assignments; with --unassign, release those with no local session (orphans)."""
import datetime, json, sys
from pathlib import Path
LEDGER = Path(__file__).resolve().parents[1] / "outputs/colab/ledger.jsonl"
from colab_cli.common import state
sessions = state.store.list()
items = sessions.values() if isinstance(sessions, dict) else sessions
local = {getattr(s, "endpoint", None) for s in items}
live = state.client.list_assignments()
orphans = [a for a in live if a.endpoint not in local]
print(json.dumps({"live": len(live), "local": len(local - {None}), "orphans": [str(a.accelerator) for a in orphans]}))
if "--unassign" in sys.argv:
    for a in orphans:
        state.client.unassign(a.endpoint)
        with open(str(LEDGER), "a") as f:
            f.write(json.dumps({"t": datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
                                "event": "unassign_orphan", "accelerator": str(a.accelerator)}) + "\n")
        print("unassigned orphan", a.accelerator)
