#!/usr/bin/env python3
"""Run inkling-dlm jobs on a Colab GPU from a shell, detached, with the session stopped as soon as the job ends.

  python scripts/colab_ctl.py up                 # balance, then a new A100 (high-RAM) session
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


def balance() -> float | None:
    r = colab("usage", capture=True)
    for line in (r.stdout or "").splitlines():
        if "balance" in line.lower():
            try:
                return float(line.split(":")[1].split()[0])
            except (IndexError, ValueError):
                pass
    return None


def ledger(event: str, **kw):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "ledger.jsonl", "a") as f:
        f.write(json.dumps(dict(t=time.strftime("%Y-%m-%dT%H:%M:%S%z"), event=event, session=SID, **kw)) + "\n")


def cmd_up(a):
    b = balance()
    print(f"balance before: {b} CU", flush=True)
    ledger("up", balance=b)
    args = ["new", "-s", SID, "--gpu", a.gpu] + (["--high-mem"] if a.high_mem else [])
    r = colab(*args, timeout=300)
    if r.returncode != 0:
        raise SystemExit(f"colab new failed ({r.returncode})")
    print(exec_code("import subprocess;print(subprocess.run(['nvidia-smi','--query-gpu=name,memory.total',"
                    "'--format=csv,noheader'],capture_output=True,text=True).stdout)", 60), flush=True)


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
sh("rm -rf {J} && git clone -q {REPO_URL} {J} && cd {J} && git checkout -q {sha} && git rev-parse HEAD")
sh("cd {J} && pip install -q -e . 2>&1 | tail -3")
sh("python -c 'import torch, transformers, peft; print(torch.__version__, transformers.__version__, peft.__version__)'")
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
            if time.time() - t0 > a.max_hours * 3600:
                print("max wait reached", flush=True)
                break
            time.sleep(a.poll)
        cmd_pull()
    finally:
        if not a.keep:
            cmd_stop()


def cmd_gates(a):
    """The first A100 stage: every GPU gate, then stop. No training."""
    cmd_up(a)
    try:
        cmd_push(a)
        a.job, a.approved = "gates", False
        cmd_launch(a)
    except BaseException:
        cmd_stop()
        raise
    cmd_wait(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["up", "push", "launch", "status", "wait", "pull", "stop", "gates"])
    ap.add_argument("job", nargs="?", default="gates")
    ap.add_argument("overrides", nargs="*")
    ap.add_argument("--gpu", default="A100")
    ap.add_argument("--high-mem", dest="high_mem", action="store_true", default=True)
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
