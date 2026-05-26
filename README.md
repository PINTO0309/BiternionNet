# BiternionNet

PyTorch + `uv` training pipeline for the experiments from "BiternionNets: Continuous Head Pose Regression from Discrete Training Labels".

The original notebooks are kept as historical reference, but the supported execution path is now Python package code under `src/biternionnet` plus CLI commands.

## Acknowledgements

This repository is a refactored PyTorch + `uv` fork of Lucas Beyer's original [BiternionNet](https://github.com/lucasb-eyer/BiternionNet) implementation. The core idea, experiment structure, original notebook workflows, dataset preparation scripts, and the Biternion / von Mises comparison setup come from that project.

Please cite and credit the original work when using this code:

```bibtex
@inproceedings{Beyer2015BiternionNets,
  author = {Lucas Beyer and Alexander Hermans and Bastian Leibe},
  title = {Biternion Nets: Continuous Head Pose Regression from Discrete Training Labels},
  booktitle = {Pattern Recognition},
  publisher = {Springer},
  series = {Lecture Notes in Computer Science},
  volume = {9358},
  pages = {157-168},
  year = {2015},
  isbn = {978-3-319-24946-9},
  doi = {10.1007/978-3-319-24947-6_13},
  ee = {http://lucasb.eyer.be/academic/biternions/biternions_gcpr15.pdf}
}
```

The changes in this fork are engineering changes around packaging, dependency management, PyTorch model/training code, JSONL manifests, and command-line execution. They are not intended to obscure or replace the authorship of the original research code.

## Setup

```bash
uv python install 3.13.11
uv sync --locked
```

Run tests:

```bash
uv run --locked pytest
```

For reproducible installs, keep `pyproject.toml` and `uv.lock` committed and use the Python version in `.python-version` with `--locked` in CI. Use `--frozen` when you want commands to fail instead of updating the lock file:

```bash
uv run --frozen pytest
```

## Manifest Format

Training uses JSONL manifests. Image paths are relative to the manifest file unless absolute.

Classification:

```json
{"split":"train","image":"images/0001.jpg","task":"classification","label":"front"}
{"split":"test","image":"images/0002.jpg","task":"classification","label":"left"}
```

Single-angle regression / Biternion:

```json
{"split":"train","image":"images/0001.jpg","task":"angle_deg","angle_deg":90.0}
{"split":"test","image":"images/0002.jpg","task":"angle_deg","angle_deg":270.0}
```

Three-axis radian pose regression:

```json
{"split":"train","image":"images/0001.jpg","task":"pose_rad","pan":0.1,"tilt":-0.2,"roll":0.0}
```

## CLI

List experiment presets:

```bash
uv run --locked biternion list-experiments
```

Train:

```bash
uv run --locked biternion-train \
  --experiment towncentre-biternion \
  --manifest data/custom/manifest.jsonl \
  --output runs/towncentre-biternion
```

Evaluate:

```bash
uv run --locked biternion-eval \
  --checkpoint runs/towncentre-biternion/best.pt \
  --manifest data/custom/manifest.jsonl
```

Convert existing Tosato JSON metadata:

```bash
uv run --locked biternion-convert \
  --source data/QMULPoseHeads.json \
  --kind tosato-classification \
  --output data/qmul/manifest.jsonl
```

## Experiment Presets

Classification presets:

- `hiit`
- `hocoffee`
- `hoc`
- `qmul`
- `qmul-no-background`

Regression / Biternion presets:

- `idiap`
- `caviar`
- `caviar-occluded`
- `towncentre-linreg`
- `towncentre-linreg-rad`
- `towncentre-mod-mae`
- `towncentre-vonmises`
- `towncentre-biternion`
- `towncentre-biternion-vonmises`

TownCentre quantized-label presets follow:

```text
towncentre-q{3,4x,4p,6x,8x,8p,10x,12x}-{softmax,linreg,linreg-vonmises,biternion,biternion-vonmises}
```

## Reproduction Scripts

Run Tosato-family manifests:

```bash
uv run --locked python scripts/run_tosato.py --output-root runs/tosato
```

Run TownCentre experiments:

```bash
uv run --locked python scripts/run_towncentre.py \
  --manifest data/towncentre/manifest.jsonl \
  --output-root runs/towncentre
```

Inspect a manifest:

```bash
uv run --locked python scripts/inspect_dataset.py data/custom/manifest.jsonl
```

## Notes

- Images are loaded with OpenCV, converted from BGR to RGB, scaled to `float32` in `[0, 1]`, cropped, and returned as `C,H,W` tensors.
- The `uv` interpreter is pinned to Python `3.13.11` in `.python-version`; package metadata allows Python `>=3.11`.
- Direct runtime dependencies and the build backend are pinned exactly in `pyproject.toml`; resolved transitive dependencies and artifact hashes are recorded in `uv.lock`.
- Checkpoints contain `model_state_dict`, `optimizer_state_dict`, the experiment config, `class_to_idx`, and metric history. Quantization borders/centres are included inside the experiment config for quantized presets.
- Numerical results are not expected to match the original Theano implementation bit-for-bit. This is a PyTorch port of the original notebook architecture, losses, metrics, and experiment presets, with framework/runtime differences.
