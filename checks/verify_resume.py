"""Kill and restart: a training process killed (SIGKILL) after a checkpoint resumes from the newest complete state
and ends bit-identical to an uninterrupted run; W&B continues the same run id."""
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]


def _train(run_dir, updates, extra=()):
    # a fresh process with its own W&B service, as after a VM restart (not the pytest process's service)
    env = {k: v for k, v in os.environ.items() if not (k.startswith("WANDB_SERVICE") or k.startswith("_WANDB"))}
    env.update(WANDB_MODE="offline", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, "-m", "inkling_dlm", "train", "--config", str(ROOT / "configs" / "cpu-tiny.yaml"),
           f"run_dir={run_dir}", f"stage_dir={run_dir}/stage", f"updates={updates}", "save_every=2",
           "eval_every=0", "max_examples=64", f"cache_dir={run_dir}/cache", "warmup=2",
           f"data_local_dir={os.environ.get('INKLING_LAD_DIR') or 'none'}", *extra]
    return subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            start_new_session=True)


def _final(run_dir):
    st = sorted(p for p in Path(run_dir).glob("state-*") if not p.name.endswith(".partial"))[-1]
    return torch.load(st / "state.pt", map_location="cpu", weights_only=False)


def check_kill_and_restart_is_exact(tmp_path):
    ref_dir, kill_dir = tmp_path / "ref", tmp_path / "killed"
    p = _train(ref_dir, 8)
    out, _ = p.communicate(timeout=900)
    assert p.returncode == 0, out[-3000:]
    p = _train(kill_dir, 8)
    t0 = time.time()
    while not (kill_dir / "state-000004").exists():
        assert p.poll() is None, p.stdout.read()[-3000:]
        assert time.time() - t0 < 900
        time.sleep(0.2)
    os.killpg(p.pid, signal.SIGKILL)                           # preemption: the whole process group (incl. the
    p.wait()                                                   # W&B service) dies, no cleanup, no final save
    done = sorted(x.name for x in kill_dir.glob("state-*") if not x.name.endswith(".partial"))
    assert done and done[-1] < "state-000008"
    wid = (kill_dir / "wandb_id").read_text()
    p = _train(kill_dir, 8)
    out, _ = p.communicate(timeout=900)
    assert p.returncode == 0, out[-3000:]
    assert "start at update 0" not in out and "start at update" in out
    assert (kill_dir / "wandb_id").read_text() == wid          # same W&B run continues
    a, b = _final(ref_dir), _final(kill_dir)
    assert a["step"] == b["step"] == 8
    for n in a["trainable"]:
        assert torch.equal(a["trainable"][n], b["trainable"][n]), n
    kept = sorted(x.name for x in kill_dir.glob("state-*"))
    assert kept == ["state-000006", "state-000008"]           # keep=2, no .partial left behind
