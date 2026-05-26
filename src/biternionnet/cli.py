from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

from .converters import (
    convert_caviar_pickle,
    convert_idiap_pickle,
    convert_pickle_classification,
    convert_tosato_classification,
    convert_towncentre_pickle,
)
from .experiments import list_experiments
from .train import evaluate_checkpoint, train_model

app = typer.Typer(help="PyTorch BiternionNet training utilities.")


@app.command("list-experiments")
def list_experiments_command() -> None:
    for name in list_experiments():
        typer.echo(name)


def train_command(
    experiment: str = typer.Option(..., help="Experiment preset name."),
    manifest: Path = typer.Option(..., exists=True, readable=True, help="JSONL manifest."),
    output: Path = typer.Option(..., help="Output run directory."),
    epochs: Optional[int] = typer.Option(None, help="Override epoch count."),
    batch_size: Optional[int] = typer.Option(None, help="Override batch size."),
    lr: Optional[float] = typer.Option(None, help="Override learning rate."),
    seed: int = typer.Option(0, help="Random seed."),
    device: Optional[str] = typer.Option(None, help="Torch device, e.g. cpu or cuda."),
    num_workers: int = typer.Option(0, help="DataLoader worker count."),
    train_flip_probability: float = typer.Option(0.5, min=0.0, max=1.0, help="Training horizontal flip probability."),
) -> None:
    result = train_model(
        experiment,
        manifest,
        output,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=seed,
        device_name=device,
        num_workers=num_workers,
        train_flip_probability=train_flip_probability,
    )
    typer.echo(json.dumps(result, sort_keys=True))


def eval_command(
    checkpoint: Path = typer.Option(..., exists=True, readable=True, help="Checkpoint path."),
    manifest: Path = typer.Option(..., exists=True, readable=True, help="JSONL manifest."),
    split: str = typer.Option("test", help="Manifest split to evaluate."),
    device: Optional[str] = typer.Option(None, help="Torch device, e.g. cpu or cuda."),
    batch_size: Optional[int] = typer.Option(None, help="Override batch size."),
    num_workers: int = typer.Option(0, help="DataLoader worker count."),
) -> None:
    evaluate_checkpoint(checkpoint, manifest, split=split, device_name=device, batch_size=batch_size, num_workers=num_workers)


def convert_command(
    source: Path = typer.Option(..., exists=True, readable=True, help="Source dataset metadata."),
    kind: str = typer.Option(..., help="tosato-classification, pickle-classification, towncentre-pickle, idiap-pickle, caviar-pickle."),
    output: Path = typer.Option(..., help="Output JSONL manifest."),
    train_root: str = typer.Option("", help="Training image root for CAVIAR-like pickles."),
    test_root: str = typer.Option("", help="Test image root for CAVIAR-like pickles."),
    image_root: str = typer.Option("TownCentreHeadImages", help="Image root for TownCentre pickle conversion."),
) -> None:
    if kind == "tosato-classification":
        convert_tosato_classification(source, output)
    elif kind == "pickle-classification":
        convert_pickle_classification(source, output)
    elif kind == "towncentre-pickle":
        convert_towncentre_pickle(source, output, image_root=image_root)
    elif kind == "idiap-pickle":
        convert_idiap_pickle(source, output)
    elif kind == "caviar-pickle":
        if not train_root or not test_root:
            raise typer.BadParameter("--train-root and --test-root are required for caviar-pickle")
        convert_caviar_pickle(source, output, train_root=train_root, test_root=test_root)
    else:
        raise typer.BadParameter(f"Unsupported conversion kind: {kind}")
    typer.echo(str(output))


@app.command("train")
def train_subcommand(
    experiment: str = typer.Option(...),
    manifest: Path = typer.Option(..., exists=True, readable=True),
    output: Path = typer.Option(...),
    epochs: Optional[int] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    lr: Optional[float] = typer.Option(None),
    seed: int = typer.Option(0),
    device: Optional[str] = typer.Option(None),
    num_workers: int = typer.Option(0),
    train_flip_probability: float = typer.Option(0.5, min=0.0, max=1.0),
) -> None:
    train_command(experiment, manifest, output, epochs, batch_size, lr, seed, device, num_workers, train_flip_probability)


@app.command("eval")
def eval_subcommand(
    checkpoint: Path = typer.Option(..., exists=True, readable=True),
    manifest: Path = typer.Option(..., exists=True, readable=True),
    split: str = typer.Option("test"),
    device: Optional[str] = typer.Option(None),
    batch_size: Optional[int] = typer.Option(None),
    num_workers: int = typer.Option(0),
) -> None:
    eval_command(checkpoint, manifest, split, device, batch_size, num_workers)


@app.command("convert")
def convert_subcommand(
    source: Path = typer.Option(..., exists=True, readable=True),
    kind: str = typer.Option(...),
    output: Path = typer.Option(...),
    train_root: str = typer.Option(""),
    test_root: str = typer.Option(""),
    image_root: str = typer.Option("TownCentreHeadImages"),
) -> None:
    convert_command(source, kind, output, train_root, test_root, image_root)


def main_train() -> None:
    typer.run(train_command)


def main_eval() -> None:
    typer.run(eval_command)


def main_convert() -> None:
    typer.run(convert_command)

