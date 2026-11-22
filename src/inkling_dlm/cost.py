"""Compute-unit projection from measured update time (Colab A100 80GB: ~8.88 CU/h, as observed on this account)."""
from __future__ import annotations


def projection(s_per_update: float, updates: int, cu_per_hour: float, eval_overhead: float = 0.03) -> dict:
    hours = s_per_update * updates / 3600 * (1 + eval_overhead)
    return dict(s_per_update=s_per_update, train_hours=hours, train_cu=hours * cu_per_hour)
