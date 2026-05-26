#!/usr/bin/env python
from __future__ import annotations

import json
from pathlib import Path

import typer

from biternionnet.onnx_export import export_checkpoint_to_onnx

app = typer.Typer(help="Export a BiternionNet checkpoint to static and dynamic-batch ONNX models.")


@app.command()
def main(
    checkpoint: Path = typer.Option(..., exists=True, readable=True, help="PyTorch checkpoint path."),
    output_dir: Path = typer.Option(Path("onnx"), help="Directory for exported ONNX files."),
    prefix: str | None = typer.Option(None, help="Output filename prefix. Defaults to the checkpoint parent directory name."),
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


if __name__ == "__main__":
    app()

