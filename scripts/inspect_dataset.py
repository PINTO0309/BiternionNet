#!/usr/bin/env python
from __future__ import annotations

from collections import Counter
from pathlib import Path

import numpy as np
import typer

from biternionnet.data import read_manifest

app = typer.Typer(help="Print lightweight manifest statistics.")


@app.command()
def main(manifest: Path = typer.Argument(..., exists=True, readable=True)) -> None:
    records = read_manifest(manifest)
    print(f"records: {len(records)}")
    print("splits:", dict(Counter(r.get("split", "") for r in records)))
    print("tasks:", dict(Counter(r.get("task", "") for r in records)))
    labels = Counter(r["label"] for r in records if "label" in r)
    if labels:
        print("labels:", dict(labels))
    angles = np.array([float(r["angle_deg"]) for r in records if "angle_deg" in r], dtype=np.float32)
    if len(angles):
        print(f"angle_deg: count={len(angles)} min={angles.min():.2f} max={angles.max():.2f} mean={angles.mean():.2f}")


if __name__ == "__main__":
    app()

