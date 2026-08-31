# history/

Ablation and experiment log for this fork. One numbered Markdown file per topic, written when the
work is done (or planned) so that later entries can refer back to earlier ones by number.

## Conventions

- File name: `NNN_short_title.md`, title line `# NNN: Title`, then `Created YYYY-MM-DD (user request: ...)`.
- Numbered sections (`## 0.`, `## 1.`, ...); `## 0.` is "how to read this entry" when the numbers need caveats.
- Every number comes from a run directory under `runs/` (`history.jsonl`, `last.pt`) or a script in `scripts/`;
  say which. Tables that compare runs are produced with `scripts/compare_runs.py --markdown` and pasted verbatim.
- Figures live in `assets/NNN/`. Run directories, checkpoints and datasets are not tracked by git.
- A planned experiment gets its entry before the runs start (commands + decision rule); the results section is
  filled in afterwards, in the same file.

## Index

| # | Title | Status |
|---|---|---|
| [001](001_fidelity_fixes.md) | Fidelity fixes to the original notebooks | done |
| [002](002_towncentre_dataset.md) | TownCentre dataset analysis | done |
| [003](003_lr_schedule_sweep.md) | Epoch budget and lr-schedule sweep (constant 1000 -> cosine 100) | done |
| [004](004_augmentation_ablation.md) | Augmentation ablation (neighbour frames / photometric / scale jitter) | R0-R4 done, follow-ups planned |
| [005](005_synthetic_fill_study.md) | Can the HRFFA synthetic heads fill the sparse TownCentre bins? | done (study, negative) |
| [006](006_synthetic_generation_plan.md) | Synthetic generation plan: images needed per pan band | quota plan; pipeline details superseded by 007 |
| [007](007_synthetic_pipeline_design.md) | TownCentre synthetic-data pipeline design and HRFFA port review | revised plan (Fable 5 review incorporated) |
| [008](008_synthetic_pipeline_implementation.md) | Synthetic TownCentre reinforcement pipeline implementation | implemented; generation not yet submitted |
| [009](009_hrffa_landmark_qa.md) | HRFFA ViT-L landmark diagnostics for synthetic QA | implemented; thresholds await Validation/Pilot calibration |
| [010](010_single_batch_tensorrt_qa.md) | Single-batch TensorRT policy for synthetic ONNX QA | implemented |
| [011](011_synthetic_validation_and_edit_cycles.md) | Synthetic Validation and QA-guided Batch edit cycles | two edit rounds complete; pitch calibration blocks Pilot |
| [012](012_object_assisted_pitch_recovery.md) | Object-assisted pitch recovery | v010 executed; pitch improved but calibration still blocked |
| [013](013_qa_policy_pan30_head_ratio_diagnostic.md) | QA policy: pan 30 deg and diagnostic-only head-height ratio | implemented; v010 re-evaluated |
| [014](014_deim_direction_bin_tolerance.md) | DEIM direction tolerance by cyclic bin distance | implemented; v010 re-evaluated |
| [015](015_fixed_deim_crop_margin.md) | Fixed DEIMv2-box crop margin | implemented at 5%; v010 sheet regenerated |
| [016](016_validation_failure_repair.md) | Validation failure repair | v013 reaches 19/19 automatic QA and valid pitch calibration |
| [017](017_composite_manifests.md) | Composite manifests: current / balanced train + enlarged test side | done |
