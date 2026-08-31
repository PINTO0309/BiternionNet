# 008: Synthetic TownCentre pipeline implementation

Created 2026-08-31 (user request: implement the approved synthetic-data plan)

## 0. Scope and current state

The design in 007 is implemented. This entry records code and validation only: no paid API request was
submitted and no synthetic image was generated during implementation.

The hash-verified QA weights were copied locally into ignored `data/models/`. A local, unsubmitted
`data/synthetic/batches/validation-v001` plan contains 19 `quality=low` requests; its remote Batch count is zero.

## 1. Implemented controls

- GPT-Image-2 Batch planning, submission, status, collection, idempotent resume, and immutable plan/input hashes.
- Fixed `gpt-image-2-2026-04-21`, `/v1/images/generations`, `quality=low`, JPEG output, and explicit request-count
  plus spend-cap approval before submission.
- Validation -> Pilot -> quota-fill approval chain. The account snapshot, sign table, human review,
  `test_profiles` protocol, and Validation usage/cost report are SHA-256 bound. Later cost projections use the
  preceding stage's actual account cost per completed image.
- DEIMv2 framing/direction QA and SixDRepNet360 pan/pitch QA. Pitch is calibrated against
  `sixd_pitch + camera_elevation`, not an eye-level absolute-pitch gate.
- Pilot freezes a per-rear-sector policy: SixD rear labels require at least 70% within 25 degrees of intent and
  no more than 10% DEIM conflicts; every other rear sector uses reviewed intent labels.
- Eye-level/low-angle generations are retained in an all-elevations manifest but do not contribute to
  high-angle quotas. Pure back views remain valid and are present in Validation.
- TownCentre-real versus synthetic crop-margin contact sheet and output resizing to the real training-crop
  size distribution.
- Exact source-quota sampler with a per-synthetic-image repetition cap (default four), starting at a 10% arm.
- Person-disjoint, double-reviewed 200--300 profile test-set finalization, 45-degree-bin metrics, per-record
  predictions, and paired person-cluster bootstrap confidence intervals.
- Checked local installation of the DEIMv2 and SixDRepNet360 ONNX assets from the HRFFA repository. Adapted
  source provenance is recorded in the corresponding file headers.

## 2. Main entry points

- `configs/synthetic_towncentre.yaml`
- `src/biternionnet/synthetic/`
- `src/biternionnet/evaluation.py`
- `biternion-synthetic`, `biternion-paired-bootstrap`, and the synthetic options on `biternion-train`

The operational command sequence is documented in the root README. Local artifacts are written under
`data/synthetic/`, which is ignored by Git.

## 3. Verification

The implementation is covered by deterministic plan, paid-submit guard, response reconciliation, pitch
calibration, eye-level retention, manifest preservation, source sampling, profile split, per-bin metric, paired
bootstrap, and training/evaluation integration tests. `uv run --locked pytest -q` passed 63 tests on
2026-08-31; the only output was four pre-existing PyTorch ONNX-export deprecation warnings.
