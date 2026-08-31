"""Measured-pan quota accounting and fail-closed top-up planning."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable

from .generate import PipelineError, circular_error_deg


def absolute_pan(angle_deg: float) -> float:
    pan = float(angle_deg) % 360.0
    return min(pan, 360.0 - pan)


def absolute_pan_bin(angle_deg: float) -> int:
    return min(180, int(math.floor((absolute_pan(angle_deg) + 5.0) / 10.0)) * 10)


def target_by_bin(config: dict[str, Any], target: str) -> dict[int, int]:
    key = {"floor_120": "floor_120_accepted", "uniform_200": "uniform_200_accepted"}.get(target)
    if key is None:
        raise PipelineError(f"unknown accepted target: {target}")
    bins = list(map(int, config["targets"]["abs_pan_bins"]))
    values = list(map(int, config["targets"][key]))
    return dict(zip(bins, values, strict=True))


def accepted_counts(rows: Iterable[dict[str, Any]]) -> dict[int, int]:
    counts: Counter[int] = Counter()
    for row in rows:
        if not row.get("pan_quality_pass"):
            continue
        if not row.get("counts_toward_high_angle_quota"):
            continue
        angle = row.get("angle_deg", row.get("estimated_pan_deg"))
        if angle is None:
            continue
        counts[absolute_pan_bin(float(angle))] += 1
    return {value: counts[value] for value in range(0, 181, 10)}


def request_yield_by_bin(rows: Iterable[dict[str, Any]]) -> dict[int, float]:
    requested: Counter[int] = Counter()
    accepted: Counter[int] = Counter()
    for row in rows:
        requested_bin = int(row["abs_pan_bin"])
        requested[requested_bin] += 1
        if row.get("pan_quality_pass") and row.get("counts_toward_high_angle_quota"):
            accepted[requested_bin] += 1
    return {
        value: accepted[value] / requested[value]
        for value in range(0, 181, 10)
        if requested[value]
    }


def top_up_plan(
    config: dict[str, Any],
    rows: Iterable[dict[str, Any]],
    target: str,
    *,
    default_yield: float = 0.5,
    minimum_yield: float = 0.05,
) -> dict[str, Any]:
    rows = list(rows)
    targets = target_by_bin(config, target)
    accepted = accepted_counts(rows)
    yields = request_yield_by_bin(rows)
    request_counts: list[int] = []
    details: list[dict[str, Any]] = []
    for value in config["targets"]["abs_pan_bins"]:
        value = int(value)
        deficit = max(0, targets[value] - accepted[value])
        observed = yields.get(value)
        planning_yield = default_yield if observed is None else max(minimum_yield, observed)
        requests = math.ceil(deficit / planning_yield) if deficit else 0
        request_counts.append(requests)
        details.append(
            {
                "abs_pan_bin": value,
                "target": targets[value],
                "accepted": accepted[value],
                "deficit": deficit,
                "observed_yield": observed,
                "planning_yield": planning_yield,
                "requests": requests,
            }
        )
    return {
        "target": target,
        "accepted_total": sum(accepted.values()),
        "remaining_accepted": sum(row["deficit"] for row in details),
        "request_total": sum(request_counts),
        "bin_counts": request_counts,
        "bins": details,
    }


def intent_pan_pass(estimated_pan_deg: float, intent_pan_deg: float, tolerance_deg: float) -> bool:
    return circular_error_deg(estimated_pan_deg, intent_pan_deg) <= float(tolerance_deg)
