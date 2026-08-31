#!/usr/bin/env python
"""Donut heatmaps (notebook cells 10-18 / paper Fig. 1) from trained checkpoints.

    uv run --locked python scripts/plot_donut.py \
        --checkpoint runs/syn-balanced/last.pt --manifest data/towncentre/manifest_balanced.jsonl \
        --split test --output runs/syn-balanced/donut_test.jpg

Pass --checkpoint multiple times to compare models side by side; the ground-truth donut is
always the leftmost panel.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from biternionnet.donut import render_donut_figure

app = typer.Typer(add_completion=False)


@app.command()
def main(
    checkpoint: list[Path] = typer.Option(..., exists=True, readable=True, help="Checkpoint(s); repeat to compare."),
    manifest: Path = typer.Option(..., exists=True, readable=True),
    split: str = typer.Option("test"),
    output: Optional[Path] = typer.Option(None, help="Output .jpg (default: <first checkpoint dir>/donut_<split>.jpg)."),
    label: list[str] = typer.Option([], help="Panel labels (same order as --checkpoint)."),
    nbins: int = typer.Option(3600, min=36),
    smooth: int = typer.Option(41, min=1, help="Cyclic gaussian window (odd)."),
    device: Optional[str] = typer.Option(None),
    num_workers: int = typer.Option(0),
) -> None:
    output = output or checkpoint[0].parent / f"donut_{split}.jpg"
    info = render_donut_figure(
        [str(c) for c in checkpoint], manifest, split, output,
        labels=list(label) or None, nbins=nbins, smooth=smooth,
        device_name=device, num_workers=num_workers,
    )
    typer.echo(json.dumps(info))


if __name__ == "__main__":
    app()
