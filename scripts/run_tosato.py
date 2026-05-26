#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import typer

from biternionnet.train import train_model

app = typer.Typer(help="Run Tosato-family experiments from JSONL manifests.")


DEFAULTS = {
    "hiit": "data/hiit/manifest.jsonl",
    "hocoffee": "data/hocoffee/manifest.jsonl",
    "hoc": "data/hoc/manifest.jsonl",
    "qmul": "data/qmul/manifest.jsonl",
    "qmul-no-background": "data/qmul/manifest.jsonl",
    "idiap": "data/idiap/manifest.jsonl",
    "caviar": "data/caviar/manifest.jsonl",
    "caviar-occluded": "data/caviar-occluded/manifest.jsonl",
}


@app.command()
def main(
    output_root: Path = typer.Option(Path("runs/tosato")),
    epochs: int | None = typer.Option(None),
    device: str | None = typer.Option(None),
    only: list[str] | None = typer.Option(None, help="Experiment names to run. Defaults to all Tosato presets."),
) -> None:
    names = only or list(DEFAULTS)
    for name in names:
        manifest = Path(DEFAULTS[name])
        train_model(name, manifest, output_root / name, epochs=epochs, device_name=device)


if __name__ == "__main__":
    app()

