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
| [006](006_synthetic_generation_plan.md) | Synthetic generation plan: images needed per pan band | plan |
