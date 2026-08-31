"""Install locally available QA model assets without committing large blobs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

from .generate import PipelineError, sha256_file


def install_model_assets(
    model_config: dict[str, Any], *, source_repo: Path, repository_root: Path
) -> list[dict[str, str]]:
    """Copy only hash-verified model assets, refusing to replace mismatched files."""
    results: list[dict[str, str]] = []
    for name, asset in model_config.items():
        source = source_repo / str(asset["source"])
        target = repository_root / str(asset["target"])
        expected = str(asset["sha256"]).lower()
        if not source.is_file():
            raise PipelineError(f"model source not found: {source}")
        actual_source = sha256_file(source)
        if actual_source != expected:
            raise PipelineError(
                f"source SHA-256 mismatch for {name}: expected {expected}, got {actual_source}"
            )
        if target.exists():
            actual_target = sha256_file(target)
            if actual_target != expected:
                raise PipelineError(f"refusing to overwrite mismatched model asset: {target}")
            results.append({"name": name, "path": str(target), "sha256": expected, "status": "present"})
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        if temporary.exists():
            raise PipelineError(f"temporary model path already exists: {temporary}")
        try:
            with source.open("rb") as input_stream, temporary.open("xb") as output_stream:
                shutil.copyfileobj(input_stream, output_stream, length=1024 * 1024)
                output_stream.flush()
                os.fsync(output_stream.fileno())
            if sha256_file(temporary) != expected:
                raise PipelineError(f"copied SHA-256 mismatch for {name}")
            temporary.replace(target)
        except Exception:
            if temporary.exists():
                temporary.unlink()
            raise
        results.append({"name": name, "path": str(target), "sha256": expected, "status": "copied"})
    return results
