"""Per-pan metrics and paired person-cluster bootstrap utilities."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import typer


def circular_absolute_error_deg(prediction: float, target: float) -> float:
    return abs((float(prediction) - float(target) + 180.0) % 360.0 - 180.0)


def circular_45_bin(angle_deg: float) -> int:
    return int(((float(angle_deg) % 360.0 + 22.5) // 45.0) % 8) * 45


def summarize_angle_errors(
    targets_deg: Iterable[float], errors_deg: Iterable[float]
) -> dict[str, float | int]:
    targets = np.asarray(list(targets_deg), dtype=np.float64)
    errors = np.asarray(list(errors_deg), dtype=np.float64)
    if targets.shape != errors.shape or not len(targets):
        raise ValueError("targets and errors must be non-empty and have equal length")
    metrics: dict[str, float | int] = {"maad_deg": float(errors.mean())}
    bin_means: list[float] = []
    for centre in range(0, 360, 45):
        mask = np.asarray([circular_45_bin(value) == centre for value in targets])
        metrics[f"bin_{centre:03d}_count"] = int(mask.sum())
        if mask.any():
            value = float(errors[mask].mean())
            metrics[f"bin_{centre:03d}_maad_deg"] = value
            bin_means.append(value)
    metrics["bin_macro_maad_deg"] = float(np.mean(bin_means))
    return metrics


def _key(row: dict[str, Any]) -> str:
    value = row.get("record_id") or row.get("custom_id") or row.get("image")
    if not value:
        raise ValueError("prediction row needs record_id, custom_id, or image")
    return str(value)


def _person(row: dict[str, Any]) -> str:
    value = row.get("person_id")
    if value:
        return str(value)
    image = Path(str(row.get("image", "")))
    if image.parent.name:
        return image.parent.name
    return _key(row)


def paired_cluster_bootstrap(
    baseline_runs: list[list[dict[str, Any]]],
    candidate_runs: list[list[dict[str, Any]]],
    *,
    resamples: int = 10_000,
    seed: int = 20260831,
) -> dict[str, Any]:
    if len(baseline_runs) != len(candidate_runs) or not baseline_runs:
        raise ValueError("baseline and candidate need the same non-zero seed count")
    per_run: list[dict[str, tuple[str, float, float]]] = []
    for baseline, candidate in zip(baseline_runs, candidate_runs, strict=True):
        baseline_by_id = {_key(row): row for row in baseline}
        candidate_by_id = {_key(row): row for row in candidate}
        if set(baseline_by_id) != set(candidate_by_id):
            raise ValueError("paired prediction files contain different record IDs")
        differences: dict[str, tuple[str, float, float]] = {}
        for record_id in baseline_by_id:
            left, right = baseline_by_id[record_id], candidate_by_id[record_id]
            target_left = float(left["target_deg"])
            target_right = float(right["target_deg"])
            if circular_absolute_error_deg(target_left, target_right) > 1e-8:
                raise ValueError(f"target mismatch for {record_id}")
            baseline_error = float(left.get("error_deg", circular_absolute_error_deg(left["prediction_deg"], target_left)))
            candidate_error = float(right.get("error_deg", circular_absolute_error_deg(right["prediction_deg"], target_left)))
            differences[record_id] = (_person(left), baseline_error - candidate_error, target_left)
        per_run.append(differences)
    ids = set(per_run[0])
    if any(set(run) != ids for run in per_run[1:]):
        raise ValueError("all matched seeds must contain the same record IDs")
    items: list[tuple[str, float, float]] = []
    for record_id in sorted(ids):
        people = {run[record_id][0] for run in per_run}
        if len(people) != 1:
            raise ValueError(f"person mismatch across seeds for {record_id}")
        items.append(
            (
                people.pop(),
                float(np.mean([run[record_id][1] for run in per_run])),
                per_run[0][record_id][2],
            )
        )
    grouped: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for person, difference, target in items:
        grouped[person].append((difference, target))
    people = sorted(grouped)
    rng = np.random.default_rng(seed)

    def interval(selected: list[tuple[str, float, float]]) -> dict[str, Any]:
        selected_groups: dict[str, list[float]] = defaultdict(list)
        for person, value, _target in selected:
            selected_groups[person].append(value)
        selected_people = sorted(selected_groups)
        observed = float(np.mean([value for _person, value, _target in selected]))
        boot = np.empty(resamples, dtype=np.float64)
        for index in range(resamples):
            sampled_people = rng.choice(selected_people, size=len(selected_people), replace=True)
            values = [value for person in sampled_people for value in selected_groups[str(person)]]
            boot[index] = float(np.mean(values))
        low, high = np.quantile(boot, [0.025, 0.975])
        return {
            "n": len(selected),
            "person_clusters": len(selected_people),
            "mean_improvement_deg": observed,
            "ci95_low_deg": float(low),
            "ci95_high_deg": float(high),
            "ci_excludes_zero_positive": bool(low > 0.0),
        }

    result: dict[str, Any] = {
        "seed_pairs": len(per_run),
        "person_clusters": len(people),
        "resamples": resamples,
        "bootstrap_seed": seed,
        "overall": interval(items),
        "bins": {},
        "profile_sides": {},
    }
    for centre in range(0, 360, 45):
        selected = [item for item in items if circular_45_bin(item[2]) == centre]
        if selected:
            result["bins"][str(centre)] = interval(selected)
    left_profile = [item for item in items if 60.0 <= item[2] % 360.0 <= 120.0]
    right_profile = [item for item in items if 240.0 <= item[2] % 360.0 <= 300.0]
    if left_profile:
        result["profile_sides"]["60_120"] = interval(left_profile)
    if right_profile:
        result["profile_sides"]["240_300"] = interval(right_profile)
    if left_profile or right_profile:
        result["profile_sides"]["combined"] = interval(left_profile + right_profile)
    return result


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def paired_bootstrap_command(
    baseline: list[Path] = typer.Option(..., exists=True, readable=True),
    candidate: list[Path] = typer.Option(..., exists=True, readable=True),
    output: Path | None = typer.Option(None),
    resamples: int = typer.Option(10_000, min=100),
    seed: int = typer.Option(20260831),
) -> None:
    result = paired_cluster_bootstrap(
        [_read_jsonl(path) for path in baseline],
        [_read_jsonl(path) for path in candidate],
        resamples=resamples,
        seed=seed,
    )
    text = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if output is None:
        typer.echo(text, nl=False)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")


def main_paired_bootstrap() -> None:
    typer.run(paired_bootstrap_command)
