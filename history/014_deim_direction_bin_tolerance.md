# 014: DEIM direction tolerance by cyclic bin distance

Created 2026-08-31 (user request: pass DEIM direction QA unless bins are separated by two or more)

## 0. Rule

The eight DEIM direction classes are ordered cyclically as `front`, `left_front`, `left_side`, `left_back`,
`back`, `right_back`, `right_side`, and `right_front`. QA now compares the detected class directly with the
expected broad class:

- distance 0 or 1 bin: pass;
- distance 2 or more bins: fail;
- unknown/missing class: fail.

This replaces the former comparison between the DEIM class centre and the exact requested pan with a fixed
45-degree threshold. The former rule could reject adjacent direction classes near a sector boundary. The new
rule also handles the `right_front`/`front` and `back` wraparound correctly.

`configs/synthetic_qa_policy_v2.yaml` records `deim_direction_max_bin_distance: 1`. Each automatic QA row
records both `direction_bin_distance` and the effective maximum, while the report and Validation calibration
bind the policy hash.

## 1. Validation v010 consequence

The +20-degree record is expected as `front` but DEIMv2 detects `left_side`; these classes are exactly two bins
apart. It therefore remains a `direction_conflict` under the requested rule. Across v010, 10 records have
distance 0, eight have distance 1, and this one record has distance 2, leaving the aggregate automatic result at
18/19. The prior artifacts were preserved under `pre_deim_bin_distance_qa/` before regeneration. No image
generation or paid API operation is involved in this QA-policy update.
