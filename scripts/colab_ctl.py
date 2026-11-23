#!/usr/bin/env python3
"""Run inkling-dlm jobs on a Colab GPU from a shell, detached, with the session stopped as soon as the job ends.

  python scripts/colab_ctl.py up                 # balance, then any A100 (40 GB is enough with packed experts)
  python scripts/colab_ctl.py push [--commit SHA]  # clone the public repo at SHA on the VM, install, upload the W&B key
  python scripts/colab_ctl.py launch gates [--config configs/a100-gates.yaml] [key=value ...]
  python scripts/colab_ctl.py status             # ALIVE/EXITED + GPU + log tail
  python scripts/colab_ctl.py wait               # poll until the job exits, pull logs/results, STOP, balance
  python scripts/colab_ctl.py stop               # stop the session now, balance
  python scripts/colab_ctl.py gates              # up -> push -> launch gates -> wait (the first A100 stage)

Secrets: the W&B key goes file -> file (`colab upload`), is read by the job at runtime, and is never printed or put in
exec'd code (the CLI keeps exec history). The main training run is launched only with `launch train --approved`.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SID = os.environ.get("INKLING_SESSION", "inkling-a100")
J = "/content/inkling-dlm"                       # repo checkout on the VM
JOB = "/content/job"                             # logs and markers
REPO_URL = "https://github.com/NarendraPatwardhan/inkling-dlm.git"
COLAB = [os.path.expanduser("~/.local/bin/colab"), "--auth", "oauth2"]
WANDB_KEY = Path(os.environ.get("WANDB_KEY_FILE", "~/.config/wandb/key")).expanduser()
HF_TOKEN = Path("~/.cache/huggingface/token").expanduser()
OUT = ROOT / "outputs" / "colab"                 # gitignored


def colab(*a, timeout=None, capture=False):
    return subprocess.run(COLAB + list(a), timeout=timeout, capture_output=capture, text=True)


def exec_code(code: str, t: int = 120) -> str:
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
        f.write(code)
        name = f.name
    try:
        r = colab("exec", "-s", SID, "-f", name, "--timeout", str(t), timeout=t + 90, capture=True)
        return (r.stdout or "") + (r.stderr or "")
    finally:
        os.unlink(name)


def _usage(field: str) -> float | None:
    r = colab("usage", capture=True)
    for line in (r.stdout or "").splitlines():
        if field in line.lower():
            try:
                return float(line.split(":")[1].split()[0].split("/")[0])
            except (IndexError, ValueError):
                pass
    return None


def balance() -> float | None:
    return _usage("balance")


def burn_rate() -> float | None:
    return _usage("usage rate")


def gpu_info() -> tuple[str, int]:
    out = exec_code("import subprocess;print('GPU=' + subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',"
                    "'--format=csv,noheader,nounits'],capture_output=True,text=True).stdout.strip())", 90)
    line = next((l for l in out.splitlines() if l.startswith("GPU=")), "GPU=unknown,0")[4:]
    name, _, mem = line.rpartition(",")
    try:
        return name.strip(), int(mem)
    except ValueError:
        return line, 0


def ledger(event: str, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), event=event, session=SID, **kw)) + "\n")


CANDIDATES = {                                   # tried in order: the standard shape, then high-RAM
    "A100": [("A100", False), ("A100", True)],
}
MIN_MIB = 39_000                                 # a 40 GB A100 reports 40960 MiB


CPY = os.environ.get("COLAB_PY", os.path.expanduser("~/.local/share/uv/tools/google-colab-cli/bin/python"))


def assign(gpu: str, hm: bool) -> int:
    """scripts/colab_assign.py: reuse/adopt/request, long read timeout, polls a timed-out POST (see its docstring)."""
    cmd = [CPY, str(ROOT / "scripts/colab_assign.py"), "--session", SID, "--gpu", gpu] + (["--high-mem"] if hm else [])
    try:
        return subprocess.run(cmd, timeout=1500).returncode       # its own bound: 600 s read + 600 s poll
    except subprocess.TimeoutExpired:
        return 5


def orphans(*flags) -> int:
    return subprocess.run([CPY, str(ROOT / "scripts/colab_orphans.py"), *flags], timeout=300).returncode


def cmd_up(a):
    """Get an A100 (40 or 80 GB). Anything smaller is released at once. A card a timed-out assign left is adopted."""
    b = balance()
    print(f"balance before: {b} CU", flush=True)
    ledger("up", balance=b)
    for gpu, hm in CANDIDATES[a.gpu]:
        if hm and not a.high_mem:
            continue
        print(f"=== trying {gpu}{' high-mem' if hm else ''} ===", flush=True)
        rc = assign(gpu, hm)
        if rc != 0:
            if SID in (colab("sessions", capture=True).stdout or ""):
                rc = 0                                            # landed anyway: use it
            else:
                ledger("assign_failed", gpu=gpu, high_mem=hm, rc=rc)
                if rc == 5:                                       # POST timed out: a card may still land; a second
                    break                                         # POST now could get two. Next try adopts it.
                continue
        orphans("--unassign", "--all")                            # we hold ours; release any extra card at once
        name, mem = gpu_info()
        rate = burn_rate()
        print(f"got {name} {mem} MiB, burn rate {rate} CU/h", flush=True)
        ledger("assigned", gpu=name, mem_mib=mem, rate=rate, high_mem=hm)
        if mem >= MIN_MIB:
            a.rate, a.gpu_name = rate, name
            return
        print(f"card under {MIN_MIB} MiB: releasing it", flush=True)
        colab("stop", "-s", SID, timeout=180)
    raise SystemExit("no A100 available")


def cmd_push(a):
    sha = a.commit or subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True,
                                     text=True).stdout.strip()
    code = f'''
import subprocess, os
def sh(c):
    r = subprocess.run(c, shell=True, capture_output=True, text=True)
    print(c[:80], "->", r.returncode); print(r.stdout[-1500:], r.stderr[-1500:])
    if r.returncode: raise SystemExit(r.returncode)
os.makedirs({JOB!r}, exist_ok=True)
try:                                           # a reused/adopted card may still run an earlier try's job
    os.killpg(int(open("{JOB}/run.pid").read()), 9)
except Exception:
    pass
sh("rm -rf {J} && git clone -q {REPO_URL} {J} && cd {J} && git checkout -q {sha} && git rev-parse HEAD")
# prefetch the pinned checkpoint while the package installs (from_pretrained then reads the HF cache)
pre = "from huggingface_hub import snapshot_download as s; s('thinkingmachines/Inkling-2-Mini', revision='b85c3cfa615fc413cc9af3379f491b963f3b586a'); print('PREFETCH DONE')"
subprocess.Popen(["python", "-c", pre], stdout=open("{JOB}/prefetch.log", "w"), stderr=subprocess.STDOUT,
                 stdin=subprocess.DEVNULL, start_new_session=True)
sh("cd {J} && pip install -q -e . 2>&1 | tail -3")
pins = subprocess.run(["python", "-c", "from inkling_dlm.constants import UNSLOTH; print(*UNSLOTH, sep=chr(10))"],
                      cwd={J!r}, capture_output=True, text=True).stdout.split(chr(10))
pins = [p for p in pins if p.strip()]
r = subprocess.run(["pip", "install", "-q", "--no-deps", *pins], capture_output=True, text=True)  # keeps transformers 5.24
print("unsloth", [p.rsplit("@", 1)[-1][:9] for p in pins], "->", r.returncode, r.stderr[-800:])
if r.returncode: raise SystemExit(r.returncode)
sh("python -c 'import torch, transformers, peft, triton; print(torch.__version__, transformers.__version__, peft.__version__, triton.__version__)'")
sh("cd {J} && python -c 'import inkling_dlm.quant as q; print(q.IMPL)'")
'''
    print(exec_code(code, 900), flush=True)
    if WANDB_KEY.is_file():
        colab("upload", "-s", SID, str(WANDB_KEY), "/content/wandb.key", timeout=120)
        print("W&B key uploaded file-to-file (not printed)", flush=True)
    else:
        raise SystemExit("no W&B key file on the box: set WANDB_KEY_FILE")
    if HF_TOKEN.is_file():
        colab("upload", "-s", SID, str(HF_TOKEN), "/content/hf.token", timeout=120)
    ledger("push", commit=sha)


def cmd_launch(a):
    job = ["python", "-m", "inkling_dlm", a.job, "--config", a.config]
    if a.approved:
        job.append("--approved")
    job += a.overrides
    code = f'''
import os, subprocess, shlex
J, JOB = {J!r}, {JOB!r}
os.makedirs(JOB, exist_ok=True)
for f in ("run.exit", "run.pid"):
    p = os.path.join(JOB, f)
    os.path.exists(p) and os.remove(p)
env = dict(os.environ, PYTHONUNBUFFERED="1", WANDB_KEY_FILE="/content/wandb.key", WANDB_DIR="/content/wandb",
           HF_HUB_ENABLE_HF_TRANSFER="0")
if os.path.exists("/content/hf.token"):
    env["HF_TOKEN_PATH"] = "/content/hf.token"
os.makedirs("/content/wandb", exist_ok=True)
cmd = {job!r}
script = f"echo $$ > {{JOB}}/run.pid; cd {{J}} && " + " ".join(shlex.quote(c) for c in cmd) + f" > {{JOB}}/run.log 2>&1; echo $? > {{JOB}}/run.exit"
subprocess.Popen(["nohup", "bash", "-c", script], env=env, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                 stderr=subprocess.DEVNULL, start_new_session=True)
print("launched", cmd)
'''
    print(exec_code(code, 60), flush=True)
    ledger("launch", job=job)


STATUS = r'''
import json, os, subprocess
JOB = "%(JOB)s"
ex = open(JOB + "/run.exit").read().strip() if os.path.exists(JOB + "/run.exit") else None
alive = False
try:
    pid = int(open(JOB + "/run.pid").read()); os.kill(pid, 0); alive = True
except Exception:
    pass
gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total",
                      "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(json.dumps({"state": "EXITED" if ex is not None else ("ALIVE" if alive else "DEAD"), "exit": ex, "gpu": gpu}))
if os.path.exists(JOB + "/run.log"):
    print("--- log ---"); print("\n".join(open(JOB + "/run.log", errors="replace").read().splitlines()[-25:]))
'''


def cmd_status(a=None) -> str:
    out = exec_code(STATUS % {"JOB": JOB}, 90)
    print(out, flush=True)
    return out


def _state(out: str):
    for line in out.splitlines():
        if line.startswith("{") and '"state"' in line:
            try:
                return json.loads(line)["state"]
            except ValueError:
                pass
    return None


def cmd_pull(a=None):
    OUT.mkdir(parents=True, exist_ok=True)
    for remote, local in [(f"{JOB}/run.log", "run.log"), (f"{JOB}/run.exit", "run.exit")]:
        colab("download", "-s", SID, remote, str(OUT / local), timeout=180)
    for rd in ("/content/inkling-dlm/runs/gates", "/content/drive/MyDrive/inkling-dlm/runs/main"):
        for f in ("gates.json", "sanity.json"):
            colab("download", "-s", SID, f"{rd}/{f}", str(OUT / f"{Path(rd).name}-{f}"), timeout=120,
                  capture=True)
    print(f"pulled -> {OUT}", flush=True)


def cmd_stop(a=None):
    colab("stop", "-s", SID, timeout=180)
    colab("sessions")
    b = balance()
    print(f"balance after: {b} CU", flush=True)
    ledger("stop", balance=b)


def cmd_wait(a):
    t0 = time.time()
    try:
        while True:
            st = _state(cmd_status())
            if st in ("EXITED", "DEAD"):
                break
            if time.time() - t0 > getattr(a, "deadline_s", a.max_hours * 3600):
                print("max wait reached", flush=True)
                break
            time.sleep(a.poll)
        cmd_pull()
    finally:
        if not a.keep:
            cmd_stop()


def cmd_gates(a):
    """The first GPU stage: every gate, then stop. No training. The whole session is held to a CU budget."""
    t_up = time.time()
    cmd_up(a)
    try:
        cmd_push(a)
        rate = max(r for r in (a.rate, burn_rate(), 8.88) if r)          # 8.88: measured A100 80 GB rate
        end = t_up + a.cu_budget / rate * 3600 - 240                     # margin for pull + stop
        print(f"budget {a.cu_budget} CU at {rate} CU/h -> job must end within {(end - time.time()) / 60:.0f} min",
              flush=True)
        ledger("budget", rate=rate, cu_budget=a.cu_budget, minutes_left=(end - time.time()) / 60)
        if end - time.time() < 600:
            raise SystemExit("not enough budget left for the gates")
        gpu_tag = "a100-40gb" if "40GB" in a.gpu_name.replace(" ", "") else "a100"
        a.job, a.approved = "gates", False
        a.overrides = list(a.overrides) + [f"cu_per_hour={rate}", f"tags=gates,{gpu_tag}"]
        cmd_launch(a)
    except BaseException:
        cmd_stop()
        raise
    a.deadline_s = max(60.0, end - time.time())
    cmd_wait(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "push", "launch", "status", "wait", "pull", "stop", "gates"])
    ap.add_argument("job", nargs="?", default="gates")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--gpu", default="A100", choices=sorted(CANDIDATES))
    ap.add_argument("--cu-budget", dest="cu_budget", type=float, default=5.0)
    ap.add_argument("--high-mem", dest="high_mem", action="store_true", default=True)
    ap.add_argument("--no-high-mem", dest="high_mem", action="store_false")
    ap.add_argument("--commit", default=None)
    ap.add_argument("--config", default="configs/a100-gates.yaml")
    ap.add_argument("--approved", action="store_true")
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-hours", dest="max_hours", type=float, default=1.0)
    ap.add_argument("--keep", action="store_true", help="do not stop the session after wait")
    a = ap.parse_args()
    {"up": cmd_up, "push": cmd_push, "launch": cmd_launch, "status": cmd_status, "wait": cmd_wait, "pull": cmd_pull,
     "stop": cmd_stop, "gates": cmd_gates}[a.cmd](a)


if __name__ == "__main__":
    main()
