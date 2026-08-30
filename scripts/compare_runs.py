#!/usr/bin/env python
"""Compare training runs on an equal-optimizer-step footing.

For each run directory (containing last.pt) print the test metric averaged over the last
``--steps`` optimizer steps (converted to epochs via the checkpoint's global_step / epochs),
its standard deviation, the minimum, and the final-epoch value.

    uv run --locked python scripts/compare_runs.py runs/aug-r0-baseline runs/aug-r1-nb3 --steps 2000
"""
from __future__ import annotations

from pathlib import Path

import json

import numpy as np
import torch
import typer


def load_history(run: Path) -> tuple[list[dict], int]:
    """Return (history, global_step) from last.pt, or from history.jsonl if no checkpoint exists."""
    path = run / "last.pt" if run.is_dir() else run
    if path.suffix == ".pt" and path.exists():
        data = torch.load(path, map_location="cpu")
        return data.get("history", []), int(data.get("global_step", 0))
    jsonl = run / "history.jsonl" if run.is_dir() else run
    history = [json.loads(line) for line in jsonl.read_text(encoding="utf-8").splitlines() if line.strip()]
    run_json = jsonl.parent / "run.json"
    steps_per_epoch = json.loads(run_json.read_text(encoding="utf-8")).get("steps_per_epoch", 1) if run_json.exists() else 1
    return history, len(history) * int(steps_per_epoch)

app = typer.Typer(add_completion=False)


@app.command()
def main(
    runs: list[Path] = typer.Argument(..., help="Run directories or checkpoint files."),
    steps: int = typer.Option(2000, min=1, help="Window length in optimizer steps for the average."),
    metric: str = typer.Option("maad_deg", help="History key to compare."),
    markdown: bool = typer.Option(False, help="Emit a Markdown table (for history/ entries)."),
) -> None:
    if markdown:
        typer.echo(f"| run | epochs | steps | win_ep | {metric} mean | std | min | final |")
        typer.echo("|---|---|---|---|---|---|---|---|")
    else:
        typer.echo(f"{'run':<32} {'epochs':>6} {'steps':>7} {'win_ep':>6} {'mean':>8} {'std':>6} {'min':>7} {'final':>7}")
    for run in runs:
        history, total_steps = load_history(run)
        if not history or metric not in history[-1]:
            typer.echo(f"{run.name:<32} (no {metric} in history)")
            continue
        values = np.array([float(r[metric]) for r in history])
        total_steps = total_steps or len(history)
        steps_per_epoch = max(1.0, total_steps / len(history))
        window = max(1, min(len(history), int(round(steps / steps_per_epoch))))
        tail = values[-window:]
        if markdown:
            typer.echo(f"| {run.name} | {len(history)} | {total_steps} | {window} | {tail.mean():.2f} | {tail.std():.2f} | {tail.min():.2f} | {values[-1]:.2f} |")
        else:
            typer.echo(f"{run.name:<32} {len(history):>6} {total_steps:>7} {window:>6} {tail.mean():>8.2f} {tail.std():>6.2f} {tail.min():>7.2f} {values[-1]:>7.2f}")


if __name__ == "__main__":
    app()
