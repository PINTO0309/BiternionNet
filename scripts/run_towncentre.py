#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import typer

from biternionnet.train import train_model

app = typer.Typer(help="Run TownCentre regression, Biternion, and quantized-label experiments.")


BASE_EXPERIMENTS = [
    "towncentre-linreg",
    "towncentre-linreg-rad",
    "towncentre-mod-mae",
    "towncentre-vonmises",
    "towncentre-biternion",
    "towncentre-biternion-vonmises",
]
QUANTIZED_PREFIXES = ["3", "4x", "4p", "6x", "8x", "8p", "10x", "12x"]
QUANTIZED_SUFFIXES = ["softmax", "linreg", "linreg-vonmises", "biternion", "biternion-vonmises"]


@app.command()
def main(
    manifest: Path = typer.Option(Path("data/towncentre/manifest.jsonl")),
    output_root: Path = typer.Option(Path("runs/towncentre")),
    epochs: int | None = typer.Option(None),
    device: str | None = typer.Option(None),
    include_quantized: bool = typer.Option(True),
    only: list[str] | None = typer.Option(None, help="Experiment names to run."),
) -> None:
    names = only or list(BASE_EXPERIMENTS)
    if include_quantized and only is None:
        names.extend(f"towncentre-q{prefix}-{suffix}" for prefix in QUANTIZED_PREFIXES for suffix in QUANTIZED_SUFFIXES)
    for name in names:
        train_model(name, manifest, output_root / name, epochs=epochs, device_name=device)


if __name__ == "__main__":
    app()

