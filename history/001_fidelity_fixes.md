# 001: Fidelity fixes to the original notebooks

Created 2026-08-30 (user request: analyse this paper implementation, then fix what deviates from the
paper / original notebooks). Commit `2401b7b`.

## 0. Scope

The repository is a PyTorch port of Lucas Beyer's Theano/DeepFried2 notebooks for *Biternion Nets:
Continuous Head Pose Regression from Discrete Training Labels* (GCPR 2015). The losses, initialisation,
optimiser (AdaDelta lr=1, rho=0.95, eps=1e-7, batch 100, 50 epochs) and the quantised-label experiment
definitions were already faithful; the deviations below were found by reading the notebooks cell by cell
(`Experiments - TownCentre.ipynb`, `Experiments - Tosato.ipynb`) and the paper (Table 3 / 4).

## 1. Deviations found and fixed

| # | Deviation | Effect | Fix |
|---|---|---|---|
| A | `MaxPool2d(2)` floors 17 -> 8; Theano pooling rounds up (17 -> 9). The 46x46 net produced a 64@4x4 map (`Linear(1024, 512)`) instead of the notebook's 64@5x5 (`Linear(1600, 512)`) | different architecture for every 46x46 experiment (TownCentre, QMUL, HIIT, HOCoffee, CAVIAR) | `ceil_mode=True` (`ModelConfig.pool_ceil_mode`, default True); checkpoints without the key load with floor pooling |
| B | No 50x50 resize before the 46x46 random crop. Raw TownCentre crops are ~25x23 px, so the image was upscaled to exactly 46x46 and the "random" crop had a single position | no crop augmentation at all; anisotropic upscaling | `ExperimentConfig.resize_size=(50, 50)` on all TownCentre presets, `CropConfig.resize` |
| C | `best.pt` selected on the test metric | test leakage in every reported number | `last.pt` always; `best.pt` only when the manifest has a `val` split (`biternion-convert --val-split`) |
| D | Quantised-label presets were scored against the bin centre, softmax presets only reported bin accuracy | Table 4 of the paper not reproducible | evaluation target is the continuous angle for all `quantized_*` presets; softmax reports `maad_deg` (class centre), `maad_quadint_deg` (quadratic interpolation, ported from the notebook) and `bin_accuracy` |
| E | Horizontal flip applied to IDIAP (`pose_rad`) without a label rule; CAVIAR flipped although the notebooks do not augment it | corrupted pan/roll labels for IDIAP | `flip_augmentation=False` for idiap / caviar / caviar-occluded; `flip_label` raises for unsupported tasks |
| F | `towncentre-vonmises` used kappa=1.0 | notebook uses `VonMisesCriterion(0.5, radians=False)` | kappa=0.5 |
| G (later, same day) | Lanczos upscaling of float images overshoots [0, 1]; the notebooks resized uint8 images, which OpenCV saturates | values > 1 fed to the network | clip after resize (`resize_for_crop`) |

Not ported (documented in README): shallow "pure linear regression", `ModuloMADCriterion` with N(0, 20)
init, DeepFried2's post-training BatchNorm statistics pass, multi-crop test-time augmentation, averaging
over five networks.

## 2. Evidence

- Backbone output before/after: `(64, 4, 4)` -> `(64, 5, 5)` for 46x46 input (`tests/test_models.py`).
- TownCentre raw crop sizes (sample of 500): median 29x28 px, 94 % smaller than 46 px on at least one side.
- The pre-fix checkpoint `runs/towncentre-biternion/best.pt` (swish backbone, 43/50 epochs, test-selected)
  reported 24.75 deg MAAD; it still evaluates to 24.748 through the legacy-pooling shim.
- ONNX export of the fixed model shows `Gemm [512, 1600]`.

## 3. Step-count equivalence to the paper

The notebooks double the training set with flipped copies (7 920 "heads" = 3 960 x 2), so one notebook
epoch is 80 steps and the paper's 50 epochs are 4 000 optimizer steps. This port flips online, so one
epoch is 40 steps and **the paper's budget corresponds to 100 epochs here** (see 003).
