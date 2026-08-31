# 011: Synthetic validation and QA-guided Batch edit cycles

Created 2026-08-31 (user request: execute Validation in order, feed QA failures back as source images with
concrete edit instructions, and ignore lower-body-only defects)

## 0. Outcome

The `gpt-image-2`, `quality=low` Validation workflow was exercised through generation and two bounded edit
rounds. Every paid request used the Batch API. No synchronous image request or silent fallback was used.

- `validation-v007`: 19 generated images; 12/19 passed automatic quality and pan QA.
- `validation-v008-edit01`: eight failed source JPEGs were embedded in `/v1/images/edits` requests; the other
  eleven images were copied byte-for-byte. The Batch completed 8/8 and automatic quality/pan yield became
  18/19.
- `validation-v009-edit02`: three remaining actionable records were edited; sixteen images were carried
  forward byte-for-byte. The Batch completed 3/3 and automatic quality/pan yield remained 18/19.
- The only remaining geometric automatic failure is the requested -70 deg record, which DEIM classifies in the
  adjacent rear sector. Further automatic editing stopped at the fixed two-round limit.

These counts are API/result counts, not account charges. Actual billing remains unverified and must be entered
through `usage-report` before Validation approval.

## 1. Edit-cycle contract

`biternion-synthetic edit-plan` now:

- requires a completed parent QA record and validates exact record identity;
- binds the parent state, plan, QA, source filename, and source SHA-256;
- embeds each failed JPEG as a base64 data URL in a Batch `/v1/images/edits` request;
- derives explicit corrections for head size, crop margin, direction/pan, camera-relative pitch, head/body
  detection, dimensions, and duplicates;
- preserves passing images by SHA-verified byte copy;
- permits missing/invalid sources to be regenerated, but does not replace a valid failed source with an
  unrelated generation;
- limits edit ancestry to two rounds and keeps each JSONL below 190 MiB;
- keeps `quality=low`, one output, JPEG q92, and the explicit request-count/spend-cap submission guard.

The spend comparison now uses decimal arithmetic, so a projection exactly equal to its cap is accepted instead
of failing from binary floating-point rounding.

## 2. QA findings

SixDRepNet360 returned folded Euler pitch near side profiles: the observed `abs_pan=90` estimates were around
147--171 deg despite usable pan estimates. Pitch calibration and elevation classification are therefore limited
to the empirically stable front/three-quarter range `abs_pan <= 60`; profiles and rear views remain unresolved
for elevation.

The final seven eligible Validation records did not establish the frozen camera-relative pitch gate:

- bias of `sixd_pitch + camera_elevation`: +22.0412 deg;
- `q90(abs(residual - bias)) + 5 deg`: 30.51208 deg;
- predeclared hard maximum: 25 deg;
- camera-elevation trend check: valid (60 deg median pitch was more negative than 30 deg).

The generated images are visibly high-angle candidates, but the numeric camera-elevation requests and SixD
pitch estimates are too dispersed for the predeclared acceptance threshold. Validation approval and paid Pilot
generation must remain blocked until the pitch-calibration design is resolved; this result must not be hidden by
raising the threshold after observing it.

## 3. Lower-body QA scope

The target region is the head, neck, both shoulders, and visible upper torso. Lower-body breakage alone is
accepted and cannot fail photorealism, framing, or anatomical-integrity review. Framing can still fail a distant
full-body composition when the head is too small, independently of lower-body quality.

The human-review column is named `head_neck_shoulders_integrity`; `body_integrity` was removed. Every prepared
review now includes `human_review_instructions.md` with the same rule. Automatic body detection is retained only
as evidence that neck/shoulder/upper-torso context exists; it is not a lower-body anatomy test.

## 4. Runtime and verification

All three ONNX models were invoked with batch size 1. DEIMv2 used CUDA then CPU and never TensorRT. SixD and
HRFFA used TensorRT, CUDA, then CPU with the ORT 1.26.0 / TensorRT 10.14.1 / CUDA 13.0 / SM86 / FP32 cache
namespace; no cache from another runtime was reused.

The final review artifacts are under `data/synthetic/batches/validation-v009-edit02/`, including the unobstructed
and landmark contact sheets, margin sheet, automatic QA JSONL/report, pitch calibration, sign table, review CSV,
and reviewer instructions. The complete test suite passed 72 tests; four pre-existing PyTorch ONNX-export
deprecation warnings remain.
