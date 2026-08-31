#!/usr/bin/env python
"""Build manifest_current.jsonl / manifest_balanced.jsonl with a shared enlarged test side.

    uv run --locked python scripts/build_composite_manifests.py \
        --combined data/towncentre/manifest_nb3_synthetic_all_elevations.jsonl \
        --source data/TownCentreHeadImages --output-dir data/towncentre

See biternionnet.balance for the semantics (per-bin neighbor quotas, synthetic holdout,
test / test_neighbor / test_synthetic splits).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from biternionnet.balance import build_composite_manifests

app = typer.Typer(add_completion=False)


@app.command()
def main(
    combined: Path = typer.Option(..., exists=True, readable=True, help="Combined manifest (anchors + neighbors + synthetic)."),
    source: Path = typer.Option(Path("data/TownCentreHeadImages"), exists=True, help="Raw TownCentre directory (for neighbour frames)."),
    output_dir: Path = typer.Option(Path("data/towncentre")),
    neighbor_cap: int = typer.Option(10, min=1, help="Maximum |frame offset| for train-balancing and test_neighbor."),
    balance_target: Optional[int] = typer.Option(None, help="Flip-effective per-10-degree-bin target (default: auto = the achievable maximum)."),
    synthetic_holdout: float = typer.Option(0.1, min=0.0, max=0.5, help="Fraction of synthetic records moved to the test_synthetic split (per-bin stratified)."),
    seed: int = typer.Option(0),
) -> None:
    summary = build_composite_manifests(
        combined,
        source,
        output_dir,
        neighbor_cap=neighbor_cap,
        balance_target=balance_target,
        synthetic_holdout=synthetic_holdout,
        seed=seed,
    )
    typer.echo(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    app()
