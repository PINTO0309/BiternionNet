#!/usr/bin/env python
"""Bar chart (JPEG) of the angle-label distribution in a JSONL manifest.

    uv run --locked python scripts/plot_angle_distribution.py data/towncentre/manifest.jsonl \
        --output data/towncentre/angle_distribution.jpg --bin-width 10

One panel per split (train / test / val, whichever exist) plus "all"; a single series per
panel, so no legend. Bins are centred so that 0 deg sits in the middle of a bin (a
``--bin-width 45`` chart therefore matches the paper's 8 canonical directions).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import typer  # noqa: E402

# Reference palette (dataviz skill): single-series blue on a light surface, text tokens for text.
SURFACE = "#fcfcfb"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
GRID = "#e4e3df"
SERIES = "#2a78d6"

app = typer.Typer(add_completion=False)


def _angles(records: list[dict], split: str | None) -> np.ndarray:
    values = [float(r["angle_deg"]) for r in records if "angle_deg" in r and (split is None or r.get("split") == split)]
    return np.mod(np.asarray(values, dtype=np.float64), 360.0)


def _histogram(angles: np.ndarray, bin_width: float) -> tuple[np.ndarray, np.ndarray]:
    """Counts per bin with bins centred on multiples of ``bin_width`` (0 deg in the middle of bin 0)."""
    n_bins = int(round(360.0 / bin_width))
    shifted = np.mod(angles + bin_width / 2.0, 360.0)
    counts, _ = np.histogram(shifted, bins=np.linspace(0.0, 360.0, n_bins + 1))
    centres = np.arange(n_bins) * bin_width
    return centres, counts


def _panel(ax, centres: np.ndarray, counts: np.ndarray, bin_width: float, title: str, subtitle: str) -> None:
    ax.set_facecolor(SURFACE)
    # thin bars with a ~2px surface gap between neighbours
    ax.bar(centres, counts, width=bin_width * 0.86, color=SERIES, linewidth=0, zorder=3)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold", color=TEXT_PRIMARY, pad=14)
    ax.text(0.0, 1.02, subtitle, transform=ax.transAxes, fontsize=9, color=TEXT_SECONDARY, ha="left", va="bottom")
    ax.set_xlim(-bin_width / 2.0, 360.0 - bin_width / 2.0)
    ticks = np.arange(0, 360, 45)
    ax.set_xticks(ticks)
    ax.set_xticklabels([f"{t}°" for t in ticks], color=TEXT_SECONDARY, fontsize=9)
    ax.set_xlabel("pan angle (0° = facing the camera, 180° = back of head)", color=TEXT_SECONDARY, fontsize=9)
    ax.set_ylabel("heads", color=TEXT_SECONDARY, fontsize=9)
    ax.tick_params(axis="y", colors=TEXT_SECONDARY, labelsize=9, length=0)
    ax.tick_params(axis="x", colors=TEXT_SECONDARY, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    # selective direct labels: the two tallest bins and the smallest non-empty bin
    order = np.argsort(counts)
    to_label = set(order[-2:].tolist())
    nonzero = [i for i in order if counts[i] > 0]
    if nonzero:
        to_label.add(nonzero[0])
    for i in to_label:
        ax.annotate(f"{int(counts[i])}", (centres[i], counts[i]), textcoords="offset points", xytext=(0, 3), ha="center", fontsize=8, color=TEXT_PRIMARY)


@app.command()
def main(
    manifest: Path = typer.Argument(..., exists=True, readable=True, help="JSONL manifest with angle_deg records."),
    output: Path | None = typer.Option(None, help="Output .jpg (default: <manifest dir>/angle_distribution.jpg)."),
    bin_width: float = typer.Option(10.0, min=1.0, max=180.0, help="Bin width in degrees (360 must be a multiple)."),
    dpi: int = typer.Option(150, min=50),
) -> None:
    if abs(360.0 / bin_width - round(360.0 / bin_width)) > 1e-9:
        raise typer.BadParameter("360 must be a multiple of --bin-width")
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
    splits = [s for s in ("train", "val", "test") if any(r.get("split") == s for r in records)]
    panels = [(None, "all")] + [(s, s) for s in splits]

    fig, axes = plt.subplots(len(panels), 1, figsize=(10, 3.2 * len(panels)), facecolor=SURFACE, squeeze=False)
    for ax, (split, label) in zip(axes[:, 0], panels):
        angles = _angles(records, split)
        centres, counts = _histogram(angles, bin_width)
        subtitle = f"{len(angles):,} heads · {int(round(360.0 / bin_width))} bins of {bin_width:g}° centred on 0°"
        _panel(ax, centres, counts, bin_width, f"Angle distribution — {label}", subtitle)
    fig.suptitle(f"{manifest}", x=0.01, ha="left", fontsize=9, color=TEXT_SECONDARY, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.985))

    output = output or manifest.parent / "angle_distribution.jpg"
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, format="jpg", facecolor=SURFACE, pil_kwargs={"quality": 92})
    typer.echo(str(output))


if __name__ == "__main__":
    app()
