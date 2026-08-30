#!/usr/bin/env python
"""Print a per-epoch table (loss, plateau rate, lr, phase, metric) from a checkpoint's history.

Use it to share training progress when deciding where to start the cosine decay:

    uv run --locked python scripts/schedule_report.py runs/x/last.pt --window 10
"""
from __future__ import annotations

from pathlib import Path

import torch
import typer

from biternionnet.schedules import plateau_rate

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: Path = typer.Argument(..., exists=True, readable=True),
    window: int = typer.Option(10, min=2, help="Window (epochs) for the loss-decrease rate column."),
    last: int | None = typer.Option(None, help="Only print the last N epochs."),
) -> None:
    data = torch.load(checkpoint, map_location="cpu")
    history = data.get("history", [])
    config = data.get("experiment", {})
    losses = [float(r["train_loss"]) for r in history]
    metric_key = next((k for k in ("maad_deg", "accuracy") if history and k in history[-1]), None)
    typer.echo(
        f"experiment={config.get('name')} lr_schedule={config.get('lr_schedule', 'constant')} "
        f"epochs={config.get('epochs')} schedule_state={data.get('schedule_state', {})}"
    )
    header = f"{'epoch':>5} {'train_loss':>11} {'rate@' + str(window):>10} {'lr':>8} {'phase':>9}"
    if metric_key:
        header += f" {metric_key:>10}"
    typer.echo(header)
    rows = list(enumerate(history, start=1))
    if last is not None:
        rows = rows[-last:]
    for i, record in rows:
        rate = plateau_rate(losses[:i], window)
        rate_text = "-" if rate is None else f"{rate:+.4f}"
        line = f"{record['epoch']:>5} {record['train_loss']:>11.5f} {rate_text:>10} {record.get('lr', config.get('lr', 1.0)):>8.4f} {record.get('phase', 'constant'):>9}"
        if metric_key:
            line += f" {record[metric_key]:>10.3f}"
        if record.get("schedule_event"):
            line += f"   <- {record['schedule_event']}"
        typer.echo(line)


if __name__ == "__main__":
    app()
