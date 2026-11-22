"""The Colab notebook is valid, current with its generator, drives only the package, gates before training, and
never trains without the approval flag."""
import ast
import re
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = ROOT / "notebooks" / "inkling_dlm_a100.ipynb"


def _code(nb):
    return [c.source for c in nb.cells if c.cell_type == "code"]


def check_notebook_is_valid_and_python_parses():
    nb = nbformat.read(NB, as_version=4)
    nbformat.validate(nb)
    for src in _code(nb):
        py = "\n".join(l[: len(l) - len(l.lstrip())] + "pass" if l.lstrip().startswith("!") else l
                       for l in src.splitlines())
        ast.parse(py)


def check_notebook_matches_generator(tmp_path):
    before = NB.read_text()
    subprocess.run([sys.executable, str(ROOT / "scripts" / "build_notebook.py")], check=True, capture_output=True)
    assert NB.read_text() == before


def check_gates_come_first_and_training_is_gated():
    src = "\n".join(_code(nbformat.read(NB, as_version=4)))
    g, t, s = src.index("inkling_dlm gates"), src.index("inkling_dlm train"), src.index("inkling_dlm sanity")
    assert g < t < s
    assert "TRAIN_APPROVED = False" in src and "if not TRAIN_APPROVED" in src
    assert 'assert g["passed"]' in src and "runtime.unassign()" in src
    assert not re.search(r"print\([^)]*key\)", src)            # the key is never printed
