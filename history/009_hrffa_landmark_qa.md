# 009: HRFFA ViT-L landmark diagnostics for synthetic QA

Created 2026-08-31 (user request: use the HRFFA ViT-L iBUG68 model to improve the synthetic pipeline and QA)

## 0. Scope

The static-batch `hrffa_vitl_ibug68_1x3x320x320.onnx` model was added as a third QA signal alongside DEIMv2
and SixDRepNet360. This change does not submit an image-generation request and does not make HRFFA an automatic
accept/reject gate. Validation and Pilot must establish thresholds against human review first.

## 1. Fixed inference contract

- Input head box: the selected DEIMv2 class-7 box on the full-resolution generated image.
- Crop: square about the box centre with side `max(box width, box height) * 1.1`, i.e. 5% of the long side on
  each edge. Areas outside the source image use a black border.
- Tensor: 320x320, BGR to RGB, `/255`, then ImageNet mean/std normalization.
- Outputs: 68 crop-normalized points and 3-class visibility logits (`outside`, `occluded`, `visible`), inverse
  transformed to the source image.

This crop is only the model's inference input. It is independent from the TownCentre materialization margin
selected by the real-versus-synthetic crop contact sheet.

## 2. Recorded diagnostics

`auto_qa.jsonl` now records raw source/crop landmark coordinates, visibility classes and probabilities,
per-part visibility summaries, high-confidence visible counts, point span, inter-eye distance, nose offset,
left/right eye visibility difference, crop/image containment ratios, the exact crop box, and an explicit
`hrffa_diagnostic_only=true` marker.

`qa_report.json` binds the model path and SHA-256 and reports aggregate medians by expected DEIM direction.
`review-prepare` also creates `landmark_contact_sheet.jpg`, keeping the original photorealism contact sheet
unobstructed. The overlay uses blue for the DEIM box, cyan for the HRFFA inference crop, and green/orange/red
for visible/occluded/outside landmarks. Each human-review row records `landmark_alignment` as `match`,
`mismatch`, or `unresolved`; approval joins those decisions to the machine metrics in a hash-bound
`landmark_calibration.json`.

No landmark signal changes `quality_gate_pass`, `pan_quality_pass_auto`, the SixD pitch classification, or the
rear label source in this revision. Hard gates may be promoted only after Pilot human review calibrates their
false-accept and false-reject behavior by direction/pan sector. HRFFA is not a sole oracle because the model was
trained using generated-image reinforcement and because iBUG68 does not cover the crown or validate rear heads.

## 3. Model asset and verification

The installer now recognizes the following Git-ignored local asset:

- target: `data/models/hrffa_vitl_ibug68_1x3x320x320.onnx`
- size: 1,231,653,123 bytes
- SHA-256: `96618ad81661dcce65eecd542b6cc3bebd1f1b9ebe0647c4a7c2dab63c4562e7`

It was copied from `/home/b920405/git/High-Angle_Robust_Fast_FaceAlignment` with source and destination hash
verification. A real CPU inference on a 33x30 TownCentre crop returned finite `(68,2)` points and `(68,3)`
visibility logits, verifying the ONNX/preprocessing/output contract. Unit and full-suite results are recorded
after final verification below.

The pre-existing unsubmitted `validation-v001` plan remains untouched, but its configuration hash predates
this model registration. `validation-v002` was the first three-model plan; after later metadata cleanup,
`validation-v003` became the current hash-valid plan. All remain unsubmitted.

## 4. Verification

- `git diff --check`: passed.
- `uv run --locked python -m compileall -q src tests`: passed.
- `uv run --locked pytest -q`: 65 passed; four pre-existing PyTorch ONNX-export deprecation warnings.
- `validation-v003`: 19 local `quality=low` requests, one shard, no remote Batch and no paid request submitted.
