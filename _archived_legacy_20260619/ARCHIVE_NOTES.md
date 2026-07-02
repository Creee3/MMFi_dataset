# Legacy Archive Notes - 2026-06-19

This directory temporarily isolates files that are not needed for the current
strict four-way MPJPE workflow. Nothing was deleted; files were only moved here
so the project root stays focused on the active experiment line.

## Active Workflow Kept In Project Root

The current active workflow is:

- Strict four-way split: `train / teacher_val / student_val / test`
- Protocol 2 with subject-disjoint evaluation
- MPJPE-optimized RGB teacher
- MPJPE-optimized LUPI/CMC student
- Bayesian tuning for the `cmc + mixup` line

Important files intentionally kept in the root:

- `config.yaml`
- `common.py`
- `mmfi_lib/mmfi.py`
- `mmfi_lib/evaluate.py`
- `train_t1_rgb_teacher_mpjpe.py`
- `train_lupi_rgb_teacher_mpjpe.py`
- `train_baseline_mpjpe.py`
- `run_bayesian_hparam_tuning_mpjpe.py`
- `run_ablation_mpjpe.py`
- `run_cmc_mixup_all_splits_mpjpe_matched_teacher.py`
- `README.md`
- `CODE_GUIDE.md`
- `conversation_notes_2026-06-16.md`

## Archived Files

### `scripts/`

Older training, comparison, debug, and non-current runner scripts:

- `check_ckpt.py`
- `common_backup.py`
- `main.py`
- `run_ablation_best_hparams.py`
- `run_bayesian_hparam_tuning.py`
- `run_cmc_mixup_all_splits.py`
- `run_cmc_mixup_all_splits_mpjpe.py`
- `run_debug_ablation.py`
- `run_temporal_cbam_cmc_comparison.py`
- `run_temporal_cbam_mixup_cmc_comparison.py`
- `run_temporal_mixup_cmc_comparison.py`
- `run_temporal_only_cmc_comparison_new.py`
- `train_baseline.py`
- `train_lupi.py`
- `train_lupi_rgb_teacher.py`
- `train_teacher.py`

### `db/`

Root-level temporary Optuna database:

- `optuna_study.db`

Current result-specific Optuna databases under `experiment_results/` were not
moved.

### `scratch/`

Temporary notes:

- `temp.md`

## Restore

To restore one archived file, move it back from this directory to the project
root. For example:

```bash
mv _archived_legacy_20260619/scripts/run_debug_ablation.py .
```

To restore every archived script:

```bash
mv _archived_legacy_20260619/scripts/*.py .
```

Use restore only when you intentionally need an older experiment branch.
