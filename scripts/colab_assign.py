"""Run with the colab CLI's own Python. Get an accelerator for session NAME without ever losing one the backend assigns.

  colab_assign.py --session inkling-a100 --gpu A100 [--high-mem] [--read-timeout 600] [--poll 600]

Order: (1) a live local session NAME is reused; (2) a live orphan of the wanted accelerator (assigned server-side, no
local session; e.g. a POST that timed out on an earlier try) is adopted; (3) a new assignment is requested with a long
read timeout (the CLI's google-auth session defaults to 120 s, shorter than an A100 cold provision). If the POST
times out, drops, or the front end answers 500/502/504 (it gave up on a slow provision), the same notebook hash is
polled (GET /tun/m/assign returns the assignment once it exists) and the assignment list is watched for a new
endpoint, until --poll seconds pass. Adoption writes the same SessionState
`colab new` writes, so every later `colab exec/upload/stop -s NAME` works on it.
exit 0 = session NAME ready (prints one JSON line), 3 = no capacity, 4 = refused (quota/entitlement/too many),
5 = POST timed out and nothing appeared yet (it may still land: the next try adopts it; do not POST again now).
"""
from __future__ import annotations

import argparse, datetime, json, sys, time, uuid
from pathlib import Path

import requests
from colab_cli.client import (Accelerator, Assignment, ColabRequestError, PostAssignmentResponse, Shape,
                              TooManyAssignmentsError)
from colab_cli.commands.session import resolve_runtime_options
from colab_cli.state import SessionState
from colab_cli.utils import get_status_code

LEDGER = Path(__file__).resolve().parents[1] / "outputs/colab/ledger.jsonl"
TRANSIENT = (requests.exceptions.Timeout, requests.exceptions.ConnectionError)


def ledger(event, **kw):
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps(dict(t=datetime.datetime.now().astimezone().strftime("%Y-%m-%dT%H:%M:%S%z"),
                                event=event, **kw)) + "\n")


def with_timeout(client, read_s):
    """Give every request of this client a (connect, read) timeout unless the caller set one."""
    orig = client.session.request

    def req(method, url, **kw):
        kw.setdefault("timeout", (30, read_s))
        return orig(method, url, **kw)
    client.session.request = req


def save(state, name, endpoint, info, variant, accelerator, shape_name, how):
    s = SessionState(name=name, token=info.token, url=info.url, endpoint=endpoint, token_expires_at=info.expires_at(),
                     variant=variant, accelerator=accelerator, machine_shape=shape_name)
    state.store.add(s)
    state.history.log_event(name, "session_created", {"endpoint": endpoint, "variant": variant,
                                                      "accelerator": accelerator, "machine_shape": shape_name,
                                                      "how": how})
    ledger("assign_" + how, session=name, accelerator=accelerator, shape=shape_name)
    print(json.dumps({"ready": name, "how": how, "accelerator": accelerator, "shape": shape_name}), flush=True)


def adopt_listed(state, name, a, how):
    save(state, name, a.endpoint, a.runtime_proxy_info, a.variant.name, a.accelerator.value, a.machine_shape.name, how)


def find_orphan(state, accel, exclude=frozenset()):
    local = {s.endpoint for s in state.store.list().values()}
    live = state.client.list_assignments()
    return next((a for a in live if a.accelerator == accel and a.endpoint not in local and a.endpoint not in exclude),
                None)


def run(state, name, gpu, high_mem, read_s, poll_s, step_s=20, sleep=time.sleep, now=time.monotonic):
    variant, accel, shape = resolve_runtime_options(gpu, None, high_mem=high_mem)
    shape_name = Shape.HIGH_RAM.name if shape == Shape.HIGH_RAM else Shape.STANDARD.name
    live = {a.endpoint: a for a in state.client.list_assignments()}
    mine = state.store.get(name)
    if mine and mine.endpoint in live:                                   # (1) still live: reuse
        if live[mine.endpoint].accelerator == accel:
            adopt_listed(state, name, live[mine.endpoint], "reused")
            return 0
        state.client.unassign(mine.endpoint)                             # wrong card under our name: release it
        ledger("unassign_wrong", session=name, accelerator=live[mine.endpoint].accelerator.value)
    if mine:
        state.store.remove(name)
    o = find_orphan(state, accel)                                        # (2) a card a timed-out POST left behind
    if o:
        adopt_listed(state, name, o, "adopted")
        return 0
    before = set(live)
    nbh, timed_out = uuid.uuid4(), False                                 # (3) request one
    print(f"[assign] requesting {accel.value} {shape_name} (read timeout {read_s}s)", flush=True)
    try:
        res = state.client.assign(nbh, variant=variant, accelerator=accel, shape=shape)
        info = res.runtime_proxy_info
        save(state, name, res.endpoint, info, variant.value, accel.value, shape_name, "new")
        return 0
    except TooManyAssignmentsError as e:
        print(f"[assign] refused (412): {str(e)[:160]}", flush=True)
        return 4
    except ColabRequestError as e:
        code = get_status_code(e)
        print(f"[assign] backend {code}: {getattr(e.response, 'reason', '')}", flush=True)
        if code == 503:                # answered: no capacity; one look for a late assignment below
            poll_s = 0
        elif code in (500, 502, 504):  # the front end gave up on a slow provision: the card may still land
            print(f"[assign] polling the assignment for up to {poll_s}s", flush=True)
            ledger("assign_timeout", session=name, accelerator=accel.value, shape=shape_name, poll_s=poll_s, code=code)
            timed_out = True
        else:
            return 4
    except TRANSIENT as e:
        print(f"[assign] POST {type(e).__name__}: polling the assignment for up to {poll_s}s", flush=True)
        ledger("assign_timeout", session=name, accelerator=accel.value, shape=shape_name, poll_s=poll_s)
        timed_out = True
    t_end = now() + poll_s
    while True:
        try:
            got = state.client._get_assignment(nbh, variant, accel, shape)
            if isinstance(got, Assignment):
                save(state, name, got.endpoint, got.runtime_proxy_info, variant.value, accel.value, shape_name,
                     "late")
                return 0
            o = find_orphan(state, accel, exclude=before)
            if o:
                adopt_listed(state, name, o, "late")
                return 0
        except (ColabRequestError, *TRANSIENT) as e:
            print(f"[assign] poll: {type(e).__name__}", flush=True)
        if now() >= t_end:
            break
        sleep(step_s)
    print(f"no {accel.value} assigned", flush=True)
    return 5 if timed_out else 3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--high-mem", action="store_true")
    ap.add_argument("--read-timeout", type=int, default=600)
    ap.add_argument("--poll", type=int, default=600)
    a = ap.parse_args()
    from colab_cli.common import state
    with_timeout(state.client, a.read_timeout)
    sys.exit(run(state, a.session, a.gpu, a.high_mem, a.read_timeout, a.poll))


if __name__ == "__main__":
    main()
