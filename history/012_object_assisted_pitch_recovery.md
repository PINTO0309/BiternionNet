# 012: Object-assisted pitch recovery

Created 2026-08-31 (user request: use a non-human reference object to guide head pitch when direct pitch
correction fails)

## 0. Decision

Keep the two-edit limit, but make the second pitch attempt materially different from the first:

1. The first `head_looks_up_at_camera` or `pitch_unusable` edit directly specifies camera elevation, head pitch,
   visible crown, upright neck, and no eye contact with the camera.
2. If the next QA reports a pitch failure again, the second edit adds exactly one small, matte, non-human object
   on the pavement near the lower image edge and instructs the person to rotate the entire head downward toward
   it. Merely moving the eyes is explicitly insufficient.

This follows the official GPT Image prompting guidance to specify a person's gaze and object interaction, and
to keep edit changes narrow while repeating invariants. The existing `gpt-image-2`, Batch-only,
`quality=low` contract is unchanged.

## 1. Deterministic prompt construction

- Object selection is deterministic by record serial: yellow tennis ball, plain red paper cup, blue beanbag, or
  orange pavement marker disc.
- Place the object near the lower edge directly below the nose at the same image x coordinate. Validation v010
  initially placed it on the signed-pan side; that changed -10 deg to -30.6 deg and +20 deg to +45.1 deg.
  Nose-aligned placement is therefore required to induce pitch without adding yaw.
- The requested downward correction is derived from `sixd_pitch - (-camera_elevation)`, rounded to 5-degree
  steps and clamped to 10--25 degrees. Missing/non-finite pitch uses 15 degrees.
- The object must occupy less than 5% of image height, remain separated from the person and outside the
  head/neck/shoulder target, and contain no text, logo, face, reflection, or person-like shape.
- Camera position/angle, person position/identity, framing, background, and pan are frozen. Only the numeric
  head-pitch instruction is overridden for recovery.
- The exact object description, position, correction, instruction, current reasons, and previous edit reasons
  are stored in `batch_state.json` and `edit_lineage.jsonl`.

## 2. Safety and status

The explicitly authorized `validation-v010-edit03-object` recovery used a third, separately recorded edit
round. Its Batch completed 3/3 requests without an API failure. Account billing remains unverified.

- -10 deg: SixD pitch improved from -19.26 to -55.33 deg; automatic quality and pan passed.
- +20 deg: SixD pitch improved from -27.01 to -58.84 deg, but under the then-current policy pan over-rotated to
  +45.10 deg, DEIM reported `left_side` instead of `front`, and head-height ratio fell just below the 0.25 gate
  at 0.24984, so this record failed. The later policy-only re-evaluation is recorded in 013.
- -70 deg: no pitch object was used because it is outside the stable `abs_pan <= 60` pitch range; direct pan
  correction changed DEIM from `right_back` to the requested `right_side`, and the record passed.
- The pitch calibration data threshold improved from 30.51208 to 26.05935 deg, but remains invalid because it
  exceeds the predeclared hard maximum of 25 deg by 1.05935 deg.

The added objects were small, visually plausible, separated from the people, and did not occlude the target
head/neck/shoulder region. No further paid edit was submitted after this result. The prompt was then revised to
the nose-aligned placement above.

Unit coverage verifies that the first pitch edit contains no reference object, a repeated second-round pitch
failure creates the deterministic nose-aligned red-cup instruction for the test record, the whole head/neck
rotation is specified, and provenance records the 25-degree correction.
