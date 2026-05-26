from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Optional

import typer

from .converters import (
    convert_caviar_pickle,
    convert_idiap_pickle,
    convert_pickle_classification,
    convert_tosato_classification,
    convert_towncentre_raw,
    convert_towncentre_pickle,
)
from .experiments import list_experiments
from .onnx_export import export_checkpoint_to_onnx
from .train import evaluate_checkpoint, train_model

app = typer.Typer(help="PyTorch BiternionNet training utilities.")
BackboneActivation = Annotated[str, typer.Option(help="Backbone activation: relu or swish.")]


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
    backbone_activation: BackboneActivation = "relu",
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
        backbone_activation=backbone_activation,
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


def export_onnx_command(
    checkpoint: Path = typer.Option(..., exists=True, readable=True, help="PyTorch checkpoint path."),
    output_dir: Path = typer.Option(Path("onnx"), help="Directory for exported ONNX files."),
    prefix: Optional[str] = typer.Option(None, help="Output filename prefix."),
    opset: int = typer.Option(17, help="ONNX opset version."),
    batch_size: int = typer.Option(1, min=1, help="Static export batch size and simplifier check batch size."),
    device: str = typer.Option("cpu", help="Torch device used for export."),
    simplify: bool = typer.Option(True, help="Run onnxsim-prebuilt optimization after export."),
) -> None:
    outputs = export_checkpoint_to_onnx(
        checkpoint,
        output_dir,
        prefix=prefix,
        opset=opset,
        batch_size=batch_size,
        device_name=device,
        simplify_models=simplify,
    )
    typer.echo(json.dumps(outputs, indent=2, sort_keys=True))


def convert_command(
    source: Path = typer.Option(..., exists=True, readable=True, help="Source dataset metadata."),
    kind: str = typer.Option(..., help="tosato-classification, pickle-classification, towncentre-raw, towncentre-pickle, idiap-pickle, caviar-pickle."),
    output: Path = typer.Option(..., help="Output JSONL manifest."),
    train_root: str = typer.Option("", help="Training image root for CAVIAR-like pickles."),
    test_root: str = typer.Option("", help="Test image root for CAVIAR-like pickles."),
    image_root: str = typer.Option("TownCentreHeadImages", help="Image root for TownCentre pickle conversion."),
    train_split: float = typer.Option(0.9, min=0.0, max=1.0, help="Train split ratio for raw TownCentre conversion."),
    seed: int = typer.Option(0, help="Random seed for raw TownCentre person-level split."),
) -> None:
    if kind == "tosato-classification":
        convert_tosato_classification(source, output)
    elif kind == "pickle-classification":
        convert_pickle_classification(source, output)
    elif kind == "towncentre-raw":
        convert_towncentre_raw(source, output, train_split=train_split, seed=seed)
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
    backbone_activation: BackboneActivation = "relu",
    seed: int = typer.Option(0),
    device: Optional[str] = typer.Option(None),
    num_workers: int = typer.Option(0),
    train_flip_probability: float = typer.Option(0.5, min=0.0, max=1.0),
) -> None:
    train_command(experiment, manifest, output, epochs, batch_size, lr, backbone_activation, seed, device, num_workers, train_flip_probability)


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


@app.command("export-onnx")
def export_onnx_subcommand(
    checkpoint: Path = typer.Option(..., exists=True, readable=True),
    output_dir: Path = typer.Option(Path("onnx")),
    prefix: Optional[str] = typer.Option(None),
    opset: int = typer.Option(17),
    batch_size: int = typer.Option(1, min=1),
    device: str = typer.Option("cpu"),
    simplify: bool = typer.Option(True),
) -> None:
    export_onnx_command(checkpoint, output_dir, prefix, opset, batch_size, device, simplify)


@app.command("convert")
def convert_subcommand(
    source: Path = typer.Option(..., exists=True, readable=True),
    kind: str = typer.Option(...),
    output: Path = typer.Option(...),
    train_root: str = typer.Option(""),
    test_root: str = typer.Option(""),
    image_root: str = typer.Option("TownCentreHeadImages"),
    train_split: float = typer.Option(0.9, min=0.0, max=1.0),
    seed: int = typer.Option(0),
) -> None:
    convert_command(source, kind, output, train_root, test_root, image_root, train_split, seed)


def main_train() -> None:
    typer.run(train_command)


def main_eval() -> None:
    typer.run(eval_command)


def main_convert() -> None:
    typer.run(convert_command)


def main_export_onnx() -> None:
    typer.run(export_onnx_command)
