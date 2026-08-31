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

Create a TownCentre manifest from the dataset extracted by `download_data.py`:

```bash
uv run --locked biternion-convert \
--source data/TownCentreHeadImages \
--kind towncentre-raw \
--output data/towncentre/manifest.jsonl \
--train-split 0.9 \
--seed 0
```

Convert existing Tosato JSON metadata:

```bash
uv run --locked biternion-convert \
--source data/QMULPoseHeads.json \
--kind tosato-classification \
--output data/qmul/manifest.jsonl
```

Train after creating a manifest:

```bash
uv run --locked biternion-train \
--experiment towncentre-biternion \
--manifest data/towncentre/manifest.jsonl \
--backbone-activation relu \
--output runs/towncentre-biternion
```
```bash
uv run --locked biternion-train \
--experiment towncentre-biternion-vonmises \
--manifest data/towncentre/manifest.jsonl \
--backbone-activation relu \
--output runs/towncentre-biternion-vonmises
```

Use `--backbone-activation swish` to replace backbone ReLU activations with Swish/SiLU for ablation runs.

The paper trains with a constant AdaDelta step for a fixed 50 epochs. Two opt-in schedules exist for ablations.

`--lr-schedule wsd` enables a Warmup-Stable-Decay schedule applied per optimizer step (linear warmup over `--warmup-fraction` of all steps, default 0.05; constant; linear decay over the last `--decay-fraction`, default 0.3, down to `lr * --final-lr-ratio`, default 0.1). The per-epoch log records the current `lr`:

```bash
uv run --locked biternion-train \
--experiment towncentre-biternion-vonmises \
--manifest data/towncentre/manifest.jsonl \
--lr-schedule wsd \
--output runs/towncentre-biternion-vonmises-wsd
```

`--lr-schedule plateau_cosine` keeps the lr constant until the training loss plateaus, then decays it with a cosine over `--cosine-epochs` epochs (default 15) to `lr * --final-lr-ratio`, and stops. The plateau detector compares the mean train loss of the first and second half of the last `--plateau-window` epochs (default 10); decay starts when that relative decrease falls below `--plateau-threshold` (default 0.02, i.e. 2%) and `epoch >= --plateau-min-epochs` (default 10). `--epochs` is the budget: if no plateau is detected, decay is forced to start at `epochs - cosine_epochs + 1`. Every epoch log carries `lr`, `phase`, `plateau_rate`, and a `schedule_event` line when the decay is scheduled.

To pick the decay point by hand while watching the logs, train with a constant lr (or `--disable-plateau-trigger`) and a generous `--epochs`, inspect the run, then resume from `last.pt` with a manual start epoch:

```bash
# phase 1: constant lr, large budget, share the logs
uv run --locked biternion-train --experiment towncentre-biternion-vonmises \
  --manifest data/towncentre/manifest.jsonl --epochs 200 --output runs/tc-btvm-manual

# inspect the loss-decrease rate
uv run --locked python scripts/schedule_report.py runs/tc-btvm-manual/last.pt --window 10

# phase 2: resume and start the cosine decay at the chosen epoch
uv run --locked biternion-train --experiment towncentre-biternion-vonmises \
  --manifest data/towncentre/manifest.jsonl --epochs 200 --output runs/tc-btvm-manual \
  --resume-from runs/tc-btvm-manual/last.pt \
  --lr-schedule plateau_cosine --decay-start-epoch 41 --cosine-epochs 15
```

`--resume-from` restores model/optimizer state, history, global step and schedule state; epoch numbering continues and schedule options on the command line override the checkpoint's.

### Text logs

Every run directory also gets plain-text logs next to the checkpoints:

- `history.jsonl` — one JSON object per epoch (the same record printed to stdout: `train_loss`, `lr`, test metrics, `phase` / `plateau_rate` for schedules, plus `time` and `epoch_seconds`). On resume it is rebuilt from the checkpoint's history and then appended to, so it always matches `last.pt`.
- `events.jsonl` — `start` / `resume` / `schedule` (decay scheduled or finished) / `finish` events with timestamps.
- `run.json` — the resolved experiment config, manifest, dataset sizes, `steps_per_epoch`, seed and device.

`scripts/compare_runs.py` and `scripts/schedule_report.py` read `last.pt`; `compare_runs.py` falls back to `history.jsonl` + `run.json` when no checkpoint is present.

### Data augmentation (opt-in)

The paper only uses a random 46x46 crop and a horizontal flip. Three optional augmentations are available for ablations; none of them changes labels except through the existing flip rule.

- **Neighbouring frames** (`biternion-convert --kind towncentre-raw --neighbor-frames 3`): TownCentre labels roughly every 100th frame, and the head turns by a median of 14 degrees per 100 frames, so the +-k unlabelled frames around every labelled *training* frame are added with the same angle (`"source": "neighbor"`, `"anchor_frame"`, `"frame_offset"` are recorded). The person split and the test records are identical to the `k=0` manifest. One epoch then has ~(2k+1)x more steps, so compare runs at equal optimizer steps (`scripts/compare_runs.py --steps 2000`).
- **Photometric** (`--photometric cctv`): brightness/contrast/gamma, short motion blur, gaussian noise, small random erasing, mild gaussian blur and JPEG re-compression, applied to the 50x50 resized image before the crop (`cctv-light` halves the probabilities; presets live in `biternionnet.augment`).
- **Scale jitter** (`--scale-jitter 0.9 1.1`): multiplies the pre-crop resize (50x50 -> 46..55) before the random 46x46 crop; the resized image is never smaller than the crop.

```bash
uv run --locked biternion-convert --source data/TownCentreHeadImages --kind towncentre-raw \
  --output data/towncentre/manifest_nb3.jsonl --train-split 0.9 --seed 0 --neighbor-frames 3

uv run --locked biternion-train --experiment towncentre-biternion-vonmises \
  --manifest data/towncentre/manifest_nb3.jsonl --epochs 160 --lr-schedule plateau_cosine \
  --disable-plateau-trigger --decay-start-epoch 146 --cosine-epochs 15 \
  --photometric cctv --scale-jitter 0.9 1.1 --output runs/aug-all
```

Evaluate:

```bash
uv run --locked biternion-eval \
  --checkpoint runs/towncentre-biternion/last.pt \
  --manifest data/towncentre/manifest.jsonl
```

### Synthetic TownCentre reinforcement data

The staged GPT-Image-2 pipeline is configured in `configs/synthetic_towncentre_batch.yaml`. Batch submission
uses the Batch-supported `gpt-image-2` alias because the dated `gpt-image-2-2026-04-21` snapshot is rejected by
the Batch API. Image quality is hard-fixed to `low`; planning is local and does not submit a paid request:

```bash
uv run --locked biternion-synthetic install-models \
  --source-repo /home/b920405/git/High-Angle_Robust_Fast_FaceAlignment
uv run --locked biternion-synthetic plan \
  --stage validation --batch-id validation-v004
```

Submission is deliberately separate and requires both the exact pending count and an explicit spend cap:

```bash
uv run --locked biternion-synthetic submit \
  --batch-dir data/synthetic/batches/validation-v004 \
  --approve-requests 19 --spend-cap-usd 0.20
```

After collection, run machine QA, prepare the human review/sign table, verify the fixed 5% DEIMv2-box crop
margin against real TownCentre crops, and record the actual account charge. Machine QA uses DEIMv2,
SixDRepNet360, and the local
HRFFA ViT-L iBUG68 model. HRFFA always receives a square 320x320 crop with 5% padding per side of the DEIM
long side; its landmark/visibility signals remain diagnostic until calibrated against Pilot human review.
SixD pitch calibration is restricted to `abs_pan <= 60`: the Validation runs showed the Euler pitch folding to
about 147--171 degrees around a 90-degree side profile, so profile/rear elevation remains unresolved.
`review-prepare` writes both an unobstructed `review_contact_sheet.jpg` and an overlaid
`landmark_contact_sheet.jpg`. Fill `landmark_alignment` with `match`, `mismatch`, or `unresolved` for every
review row; genuine rear views may be unresolved. Human integrity review is deliberately limited to the head,
neck, shoulders, and visible upper torso. Lower-body artifacts alone are acceptable and must not reject an
otherwise usable head-and-surroundings crop. Approval writes a hash-bound `landmark_calibration.json`,
but it does not activate an HRFFA rejection gate. Validation approval also binds the profile-evaluation
protocol and the account-verified Batch model identifier by SHA-256:

All three ONNX QA models run at batch size 1. SixD and HRFFA use TensorRT, CUDA, then CPU. **DEIMv2 never uses
TensorRT** because its TensorRT result has shown an unexplained accuracy regression; DEIM uses CUDA, then CPU.
Use `qa --cpu` only to force every model to CPU. The DEIM graph has a symbolic public batch axis, but the
pipeline still invokes it one image at a time and rejects any internal inference tensor whose leading dimension
is not 1. Do not replace the HRFFA model with its `Nx3x320x320` sibling or add batched DEIM calls. TensorRT
engine/timing caches for SixD and HRFFA are isolated under `data/models/trt_cache/` by ONNX Runtime, TensorRT,
CUDA runtime, GPU compute capability, precision, model SHA-256, and `batch1`. Any runtime change therefore
selects a new empty cache and can never reuse an engine produced by the previous runtime.

After `qa`, failed records enter two hash-bound Batch image-edit rounds by default; an explicitly bounded
Validation recovery may raise the recorded maximum to at most eight rounds. The edit request embeds the
failed JPEG and gives concrete corrections derived from its framing, direction, pan, and elevation failures;
records that passed are carried forward byte-for-byte. The pipeline never silently falls back to a synchronous
image request. Pitch uses a bounded two-stage correction: the first edit directly corrects the head/neck pose;
if the next QA reports the same pitch failure, the second edit adds one small deterministic pavement object near
the lower edge and makes the whole head—not only the eyes—look down toward it. The object stays outside the
head/neck/shoulder target region, preserves the requested pan and camera, and is recorded in edit lineage. The
planning cost is deliberately supplied from current account evidence and remains subject to the exact
request-count and spend-cap guard:

```bash
uv run --locked biternion-synthetic edit-plan \
  --parent-batch-dir data/synthetic/batches/VALIDATION_RUN \
  --batch-id VALIDATION_EDIT_RUN --max-edit-rounds 2 \
  --planning-cost-per-request-usd CONSERVATIVE_ACCOUNT_ESTIMATE
uv run --locked biternion-synthetic submit \
  --batch-dir data/synthetic/batches/VALIDATION_EDIT_RUN \
  --approve-requests EXACT_EDIT_COUNT --spend-cap-usd EXPLICIT_CAP
```

When Validation pitch calibration fails even though its tail records pass individual QA, add
`--include-pitch-calibration-tail` to `edit-plan`. The planner hash-binds the failed calibration and selects the
two centred residuals controlling the small-sample q90 tail. A later direction correction uses the current
SixD pan to express an explicit relative left/right rotation, preventing an edit from turning farther in the
already-wrong direction. The usage report distinguishes paid `completed_requests` from all carried-forward
`completed_images` and uses automatic QA until human-approved annotations exist.

```bash
uv run --locked biternion-synthetic collect --batch-dir data/synthetic/batches/validation-v004
uv run --locked biternion-synthetic qa --batch-dir data/synthetic/batches/validation-v004
uv run --locked biternion-synthetic review-prepare --batch-dir data/synthetic/batches/validation-v004
uv run --locked biternion-synthetic margin-sheet \
  --batch-dir data/synthetic/batches/validation-v004 \
  --output data/synthetic/batches/validation-v004/margin_sheet.jpg
uv run --locked biternion-synthetic usage-report \
  --batch-dir data/synthetic/batches/validation-v004 --actual-cost-usd ACTUAL_ACCOUNT_CHARGE
uv run --locked biternion-synthetic approve \
  --batch-dir data/synthetic/batches/validation-v004 --reviewer REVIEWER \
  --approve-sign-calibration \
  --evaluation-protocol data/towncentre/test_profiles_protocol.json \
  --usage-report data/synthetic/batches/validation-v004/usage_report.json \
  --account-verified-snapshot gpt-image-2
```

Automatic QA defaults to the separately hash-bound
`configs/synthetic_qa_policy_v2.yaml`: requested pan may differ by at most 30 degrees. DEIMv2 still measures
`head_height_ratio`, but that measurement is diagnostic only and never adds `head_too_small` or
`head_too_large` to the rejection reasons. DEIMv2 direction uses the cyclic eight-direction classes: the
detected and expected classes may differ by zero or one bin, while a distance of two or more bins is rejected.
The final head crop expands the DEIMv2 box by exactly 5% on each box axis; approval no longer accepts a crop
margin argument, and materialization rejects any approval carrying a different value. Each `auto_qa.jsonl` row
records the effective pan tolerance, height-gate state, DEIM bin distance, and permitted maximum;
`qa_report.json` and Validation `pitch_calibration.json` bind the policy path and SHA-256. This keeps completed
Batch generation configs immutable while making the current acceptance policy explicit and auditable.

Pilot planning refuses to proceed without this approved Validation evidence. Materialization writes a
high-angle-only training manifest and an all-elevations ablation manifest. Correctly labelled eye-level or
low-angle images are retained in the latter and never counted toward high-angle quotas:

```bash
uv run --locked biternion-synthetic materialize \
  --batch-dir data/synthetic/batches/pilot-v001
uv run --locked biternion-train --experiment towncentre-biternion-vonmises-aug \
  --manifest data/towncentre/manifest_nb3_synthetic.jsonl \
  --synthetic-fraction 0.10 --epoch-samples 26897 --synthetic-max-repeats 4 \
  --output runs/synthetic-pilot-10pct
```

For `floor_120` and `uniform_200`, pass both the approved Validation
`pitch_calibration.json` and approved Pilot `rear_label_policy.json` to `biternion-synthetic qa` via
`--calibration` and `--rear-policy`. The latter freezes the per-rear-sector 70% SixD agreement / 10% DEIM
conflict decision; sectors that do not qualify continue to use reviewed intent labels.

Use `biternion-synthetic profiles-plan` / `profiles-finalize` to build the person-disjoint 200--300-image
`test_profiles` set before the Pilot, and `biternion-eval --predictions-output ...` plus
`biternion-paired-bootstrap` for matched-seed, paired person-cluster confidence intervals. Promotion requires
a positive 95% CI; there is no fixed 0.5-degree target.

Training always writes `last.pt` (the original notebooks train a fixed 50 epochs and evaluate the final model). `best.pt` is written only when the manifest contains a `val` split, so checkpoint selection never looks at the test split. `biternion-convert --kind towncentre-raw --val-split 0.5` carves a person-level `val` split out of the non-train persons.

Export a checkpoint to ONNX with opset 17 and optimize it with `onnxsim-prebuilt`:

```bash
uv run --locked biternion-export-onnx \
--checkpoint runs/towncentre-biternion/last.pt \
--output-dir runs/towncentre-biternion/ \
--opset 17
```

This writes static and dynamic-batch ONNX files, plus `_sim.onnx` optimized versions. The model uses flatten + Linear layers, so spatial dimensions are fixed by the checkpoint experiment; the dynamic ONNX export intentionally makes only the batch axis dynamic.

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
- `towncentre-biternion-vonmises-aug` — the fork's current default (not a paper setting): photometric `cctv` + scale jitter, 300 constant epochs + 50 cosine epochs, meant for the neighbour-frame manifest (`--neighbor-frames 3`, 269 steps/epoch); see `history/004`
- `towncentre-biternion-long` / `towncentre-biternion-vonmises-long` — not a paper setting: 1000 epochs at the constant AdaDelta step followed by a 100-epoch cosine decay (`plateau_cosine` with a manual `decay_start_epoch`), chosen from a 1000-epoch constant-lr sweep in which test MAAD kept improving slowly without overfitting

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

Plot the angle-label distribution as a JPEG bar chart (one panel per split plus "all"; bins are centred on 0°, so `--bin-width 45` gives the paper's 8 canonical directions):

```bash
uv run --locked python scripts/plot_angle_distribution.py data/towncentre/manifest.jsonl \
  --output data/towncentre/angle_distribution.jpg --bin-width 10
```

## Fidelity to the original notebooks

- Max-pooling uses `ceil_mode=True` so the 46x46 network produces the notebook's `64@5x5` feature map (`Linear(1600, 512)`); Theano's pooling rounds partial windows up while PyTorch floors by default. Checkpoints written before this change (no `pool_ceil_mode` key) are loaded with floor pooling for compatibility.
- TownCentre presets resize every head crop to 50x50 before the 46x46 random crop (`resize_size`), matching `prepare_data.scale_all`. Raw TownCentre crops are ~25x23, so without the resize there was no crop augmentation at all.
- `idiap`, `caviar`, and `caviar-occluded` disable horizontal-flip augmentation (`flip_augmentation=False`) like the notebooks; there is no flip rule for pan/tilt/roll labels.
- `towncentre-vonmises` uses `kappa=0.5` like the notebook; the quantized `linreg-vonmises` presets keep `kappa=1.0`.
- Quantized-label presets (`towncentre-q*`) are trained on bin labels but evaluated against the continuous ground-truth angle, as in Section 5 of the paper. Softmax presets report `maad_deg` (class-centre prediction), `maad_quadint_deg` (quadratic interpolation between the best bin and its neighbours), and `bin_accuracy`.
- Not ported: the shallow "pure linear regression" baseline, the `ModuloMADCriterion` run with `N(0, 20)` last-layer init, DeepFried2's post-training BatchNorm statistics pass, multi-crop test-time augmentation, and averaging over five independently trained networks.

## Experiment history

`history/` holds one numbered Markdown entry per topic (fidelity fixes, dataset analysis, schedule sweep,
augmentation ablation, ...), with the run names, commands, result tables (`scripts/compare_runs.py --markdown`)
and figures under `history/assets/NNN/`. See `history/README.md` for the conventions and the index.

## Notes

- Images are loaded with OpenCV, converted from BGR to RGB, scaled to `float32` in `[0, 1]`, optionally resized to the preset's `resize_size`, resized if still smaller than the requested crop, cropped, and returned as `C,H,W` tensors.
- The `uv` interpreter is pinned to Python `3.13.11` in `.python-version`; package metadata allows Python `>=3.11`.
- Direct runtime dependencies and the build backend are pinned exactly in `pyproject.toml`; resolved transitive dependencies and artifact hashes are recorded in `uv.lock`.
- Checkpoints contain `model_state_dict`, `optimizer_state_dict`, the experiment config, `class_to_idx`, and metric history. Quantization borders/centres, `resize_size`, `flip_augmentation`, and `pool_ceil_mode` are included inside the experiment config.
- Numerical results are not expected to match the original Theano implementation bit-for-bit. This is a PyTorch port of the original notebook architecture, losses, metrics, and experiment presets, with framework/runtime differences.
