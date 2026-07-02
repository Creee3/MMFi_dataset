# Strict MPJPE Improvement Ideas

Date: 2026-06-19

## Current Observation

Under the new strict four-way split, the current `CMC + mixup` student result is:

```text
student_val best MPJPE: about 200.4 mm
final test MPJPE:       about 246.0 mm
final test PA-MPJPE:    about 94.2 mm
```

The gap between validation MPJPE and final test MPJPE suggests a cross-subject
generalization problem. The relatively stable PA-MPJPE suggests the model still
learns pose structure, while the large MPJPE drop is more likely caused by
absolute coordinate, root position, scale, or subject/domain shift.

Updated findings:

```text
WiFi baseline:                         MPJPE 242.2 mm | PA-MPJPE 97.2 mm
Original CMC + mixup:                  MPJPE 246.0 mm | PA-MPJPE 94.2 mm
CMC no mixup:                          MPJPE 250.6 mm | PA-MPJPE 97.5 mm
CMC + root/relative auxiliary loss:    MPJPE 247.3 mm | PA-MPJPE 95.8 mm
Baseline + root/relative auxiliary:    MPJPE 250.3 mm | PA-MPJPE 96.0 mm
Conservative CMC schedule:             MPJPE 242.6 mm | PA-MPJPE 95.1 mm
Baseline grid best so far:             MPJPE 234.48 mm | PA-MPJPE 94.56 mm
  setting: lr=7e-5, batch_size=64, mixup_alpha=0.4, dropout=0.15
```

Interpretation:

- Strong CMC hurts strict-test MPJPE.
- CMC can improve PA-MPJPE, so it helps pose structure more than absolute
  coordinate prediction.
- Root/relative auxiliary losses improve validation but do not improve strict
  final test.
- Conservative CMC scheduling nearly recovers baseline MPJPE while keeping part
  of the PA-MPJPE gain.
- A stable 230 mm target likely requires first improving the WiFi baseline
  under strict final test, not only tuning CMC.
- Baseline tuning already improves MPJPE by about 7.7 mm over the initial
  strict baseline, so the 230 mm target is plausible with a narrow follow-up
  search around `lr=7e-5`, `mixup_alpha=0.4`, and `dropout=0.15`.
- Later re-runs did not reproduce the deleted `234.48 mm` run. Treat it as an
  unreliable historical hint, not as the current strict baseline. Recent
  reproducible runs around `lr=7e-5`, `dropout=0.15`, `ly2/ff1024/head512`
  gave final-test MPJPE around `240.9-249.6 mm`, despite much lower validation
  MPJPE. The next priority is reducing validation-to-test generalization gap.
- Latest stability-grid run with weak WiFi augmentation and small mixup
  further only reached `242.25 mm` on final test. That is only a marginal
  improvement over `242.68 mm`, so this branch appears to be on a plateau.
  In that run, `mixup_alpha=0.2` was the best among the tested values, while
  `lambda_rel_pose=0.2` and `subject_robust_weight=0.1` did not produce a
  consistent win.

Planned next step:

- Keep the strict four-way split.
- Stop expanding the weak augmentation grid.
- Add checkpoint stabilization with SWA/EMA-style averaging.
- Fix dataloader worker handling so `num_workers=0` is respected.
- Compare `best.pth`, `last.pth`, and `best_swa.pth` on final test.

2026-06-25 update:

- The weak augmentation / low-mixup branch hit a plateau around 242 mm final
  MPJPE. Treat this as a useful baseline, not as evidence that the 235 mm target
  is impossible.
- The immediate code path now focuses on checkpoint stability rather than a
  broader hyperparameter grid:
  - `train_baseline_mpjpe.py` supports baseline-only `--swa_start`.
  - SWA checkpoints are saved as `best_swa.pth` for inspection.
  - `best.pth` is always the validation-selected checkpoint, whether regular or
    SWA. Final-test evaluation should report this checkpoint, avoiding accidental
    test-time preference for SWA when it did not improve validation.
  - `final_test.json` records `checkpoint` and `checkpoint_source`.
- This does not require preprocessing or duplicating the dataset, which keeps
  disk usage low on the Windows D drive.

2026-06-25 second update:

- The first checkpoint-stability run improved the strict final-test baseline to:

  ```text
  MPJPE:    238.93 mm
  PA-MPJPE: 95.97 mm
  setting:  lr=7e-5, batch_size=64, mixup_alpha=0.2, dropout=0.15,
            aug_noise=0.02, aug_freq_mask=0.05, aug_time_mask=0.05,
            lambda_rel_pose=0.0, lambda_root=0.0,
            subject_robust_weight=0.0
  checkpoint: best.pth, checkpoint_source=regular
  ```

- `subject_robust_weight=0.1` was worse for both tested `mixup_alpha` values, so
  subject-robust loss should be disabled for the next narrow search.
- SWA did not beat the regular validation-selected checkpoint in these runs.
  Keep `best_swa.pth` only as a diagnostic checkpoint for now.
- The next search should be narrow and disk-light: keep the same augmentation
  and architecture, search around `mixup_alpha=0.2`, and vary only one or two
  high-signal knobs such as dropout and learning rate.

2026-06-27 graph-head update:

- Replacing the plain MLP pose head with a lightweight topology-aware
  `graph_root` head is useful, but the gain is sensitive to mixup and bone loss.
- The best graph-head result so far is:

  ```text
  MPJPE:    239.79 mm
  PA-MPJPE: 98.33 mm
  setting:  lr=7e-5, batch_size=64, mixup_alpha=0.18, dropout=0.15,
            aug_noise=0.02, aug_freq_mask=0.05, aug_time_mask=0.05,
            pose_head_type=graph_root, lambda_bone=0.005
  ```

- Larger bone losses are not safe:

  ```text
  alpha=0.18, bone=0.005 -> 239.79 mm
  alpha=0.18, bone=0.010 -> 258.73 mm
  alpha=0.18, bone=0.020 -> 248.58 mm
  alpha=0.20, bone=0.005 -> 252.25 mm
  alpha=0.20, bone=0.010 -> 244.79 mm
  alpha=0.20, bone=0.020 -> 242.55 mm
  ```

- Interpretation:
  - `graph_root` should be kept as the main model-improvement branch.
  - `root` without graph topology should be dropped.
  - Bone loss should stay very weak. The current useful range appears to be
    `0.0-0.005`, not `0.01-0.02`.
  - The next run should search only around lower mixup and weak bone:
    `mixup_alpha in {0.14, 0.16, 0.18}` and
    `lambda_bone in {0.0, 0.0025, 0.005}`.

2026-07-01 short-path focus update:

- After switching to short hashed run directories, the focused graph-head run
  completed without the previous Windows path-length failures.
- Best focused result:

  ```text
  MPJPE:    239.08 mm
  PA-MPJPE: 99.59 mm
  setting:  lr=7e-5, batch_size=64, mixup_alpha=0.16, dropout=0.15,
            aug_noise=0.02, aug_freq_mask=0.05, aug_time_mask=0.05,
            pose_head_type=graph_root, lambda_bone=0.005
  run_dir:  experiment_results/gh_focus/r001_2f56d7c7/seed_00
  ```

- Other focused results:

  ```text
  alpha=0.16, bone=0.0025 -> 256.03 mm
  alpha=0.16, bone=0.0050 -> 239.08 mm
  alpha=0.18, bone=0.0025 -> 266.84 mm
  alpha=0.18, bone=0.0050 -> 241.90 mm
  ```

- Interpretation:
  - The best graph-head setting improved from `239.79 mm` to `239.08 mm`, but
    still does not clearly beat the historical plain-head best `238.93 mm`.
  - `lambda_bone=0.0025` is not useful in this branch; keep `0.005` if using
    `graph_root`.
  - Further tiny alpha/bone searches are unlikely to close the gap to 235 mm by
    themselves. The next high-value step is checkpoint selection/stabilization
    or a more reliable validation protocol.

2026-07-01 checkpoint-soup update:

- Added a checkpoint-stability branch for the current best graph-head setting.
  This does not change the dataset split or add another model structure.
- `train_baseline_mpjpe.py` can now save the best K validation checkpoints and
  build `topk_soup.pth` by averaging their weights:

  ```text
  --topk_ckpts 3 --topk_soup
  ```

- If the soup checkpoint has better validation MPJPE than the normal best
  checkpoint, it replaces `best.pth`; otherwise it is kept separately for
  diagnosis.
- `evaluate_final_test_checkpoints_mpjpe.py` now compares `best.pth`,
  `topk_soup.pth`, `best_swa.pth`, and `last.pth` when available.
- The small disk-light run command is saved as:

  ```text
  run_gh_soup_short.bat
  ```

- This is meant to test whether the 239 mm plateau is partly caused by noisy
  checkpoint selection. It should be interpreted together with validation MPJPE
  and final-test checkpoint comparison, not as another broad hyperparameter
  grid.

2026-07-01 tail-soup update:

- The first top-k soup run showed that better student-val MPJPE did not transfer
  to final test:

  ```text
  best.pth / topk_soup.pth -> final-test MPJPE 244.2 mm
  last.pth                 -> final-test MPJPE 241.4 mm
  ```

- This suggests the validation-minimum checkpoint selection rule is too
  aggressive for the current strict split. The next test is `tail_soup.pth`,
  which averages the final training checkpoints instead of the validation-best
  checkpoints.
- Added:

  ```text
  --tail_ckpts 5 --tail_soup
  run_gh_tail_short.bat
  ```

- `tail_soup.pth` is saved for diagnosis and is not allowed to overwrite
  `best.pth` by default. This keeps the checkpoint comparison honest and makes
  it clear whether late-stage weights generalize better than val-selected
  weights.

2026-07-02 tail-soup result:

- The automated tail-soup run completed successfully:

  ```text
  run_dir: experiment_results/gh_tail/r000_36c6907f/seed_00

  best.pth:      final-test MPJPE 247.5 mm, PA 97.6 mm
  tail_soup.pth: final-test MPJPE 239.0 mm, PA 98.0 mm
  last.pth:      final-test MPJPE 239.1 mm, PA 98.0 mm
  ```

- This strongly supports the diagnosis that single-epoch student-val selection
  is unreliable under the strict four-way split. The validation-best checkpoint
  reached `199.9 mm` on student-val but generalized worse than both late-stage
  checkpoints on final test.
- For future strict runs, treat `tail_soup.pth` and `last.pth` as diagnostics
  and predefine a late-stage checkpoint rule before looking at final-test
  metrics. Do not select among checkpoints by final-test performance in the
  final paper protocol.

## Beyond Hyperparameter Tuning

The current bottleneck is no longer ordinary hyperparameter tuning. Validation
MPJPE can reach roughly 196-202 mm, while strict final-test MPJPE remains around
239-252 mm. This points to cross-subject / cross-domain generalization and
absolute-coordinate transfer rather than model capacity alone.

Priority ideas:

1. Better diagnosis before more search.

   - Run final-test breakdown by subject, scene, and action for the current
     238.93 mm checkpoint.
   - Compare `best.pth`, `last.pth`, and any averaged checkpoint when available.
   - Check whether the final-test error is dominated by one scene/subject pair
     or by a global root/translation bias.

2. More reliable validation for checkpoint selection.

   - The current student validation set has only four subjects, so it can select
     checkpoints that look good on validation but transfer poorly to final test.
   - For baseline-only experiments, evaluate/model-select on the union of
     `teacher_val` and `student_val` while keeping the final test untouched.
   - For final reporting, after choosing hyperparameters/methods, consider a
     final refit on all non-test subjects (the 32-subject DT-Pose-style training
     set) and test only once on `S05 S10 ... S40`.

3. Architecture aligned with the observed failure.

   - PA-MPJPE is much lower and more stable than MPJPE, so pose shape is learned
     better than absolute translation/root position.
   - Replace the single absolute pose head with a two-head predictor:

     ```text
     feature -> root-relative pose head
             -> root/global position head
     final pose = relative pose + root position
     ```

   - Train with absolute MPJPE plus relative-pose and root-position supervision,
     but make the separation architectural rather than only adding auxiliary
     losses to the old head.

4. Teacher/CMC redesign instead of stronger CMC.

   - Previous CMC helped PA-MPJPE more than MPJPE, so it should be treated as a
     shape/structure regularizer rather than an absolute-coordinate teacher.
   - Try RGB teacher distillation only on root-relative pose, while the WiFi
     student still learns absolute root from ground truth.
   - Keep CMC very weak or delayed; avoid strong feature alignment that may erase
     WiFi cues needed for absolute position.

5. Domain generalization in the WiFi encoder.

   - Add lightweight subject/scene-invariant regularization such as CORAL/MMD,
     MixStyle, or a gradient-reversal subject classifier.
   - Prefer small weights and diagnostic runs, because the model still needs to
     keep location-sensitive information for absolute coordinates.

6. Checkpoint and prediction stabilization.

   - SWA over late epochs did not improve validation in the first runs.
   - A top-k checkpoint soup or EMA may be better than plain late-epoch SWA:
     save the best 3-5 validation checkpoints and average only those weights.
   - Ensembling 2-3 independently trained good checkpoints is a pragmatic way to
     test whether variance alone can close the remaining 3-5 mm gap to 235 mm.

Recommended next code direction:

- Stop broad alpha/dropout search after the current small stage-2 run.
- First implement diagnostics and safer evaluation.
- Then implement either:
  - combined-dev validation / final refit, if the goal is a defensible 235 mm
    number quickly; or
  - root-relative pose head + root head, if the goal is a real model-method
    improvement for the thesis.

## Split Protocol Notes

The current strict four-way split is intentionally different from the standard
DT-Pose/MMFi cross-subject setting.

Standard DT-Pose `protocol2-s2` uses:

```text
train: S01 S02 S03 S04 S06 S07 S08 S09
       S11 S12 S13 S14 S16 S17 S18 S19
       S21 S22 S23 S24 S26 S27 S28 S29
       S31 S32 S33 S34 S36 S37 S38 S39

eval:  S05 S10 S15 S20 S25 S30 S35 S40
```

The current strict four-way split uses:

```text
train:       S01 S02 S03 S06 S07 S08
             S11 S12 S13 S16 S17 S18
             S21 S22 S23 S26 S27 S28
             S31 S32 S33 S36 S37 S38

teacher_val: S04 S14 S24 S34
student_val: S09 S19 S29 S39
test:        S05 S10 S15 S20 S25 S30 S35 S40
```

Reasoning:

- The four-way split is more conservative for a LUPI/teacher/CMC method because
  teacher selection, student/CMC tuning, and final testing are separated.
- The final test set keeps the same 8 subjects as standard cross-subject
  evaluation, so the reported test set is not unusually small or unusually
  large. It is roughly 20% of all 40 subjects.
- The cost is that the student trains on 24 subjects instead of the standard 32,
  so strict four-way MPJPE is expected to be higher than DT-Pose-style
  cross-subject MPJPE.
- Therefore, strict four-way results should be used for method validation and
  leakage control, while standard cross-subject results should be used for fair
  numerical comparison with DT-Pose, MetaFi, HPELI, and similar baselines.

## Immediate Experiments

These are the most useful checks before changing the model.

1. Run strict WiFi baseline.

   Purpose: determine whether `246 mm` is actually poor relative to the strict
   baseline, or only poor relative to the old non-strict validation result.

2. Run strict `CMC no mixup`.

   Purpose: check whether mixup hurts absolute coordinate regression under the
   strict subject-disjoint test.

3. Try lower mixup alpha values.

   Candidate values:

   ```text
   alpha = 0.0, 0.1, 0.2, 0.4
   ```

   Reason: mixup can improve classification/generalization, but for absolute
   3D coordinate regression it may create unrealistic mixed positions.

4. Rerun Bayesian tuning under strict four-way split.

   Search at least:

   ```text
   lr
   batch_size
   mixup_alpha
   lambda_cmc
   dropout
   ```

   If compute is limited, first tune only `CMC + mixup`.

## Model-Level Ideas

### 1. Root-Relative Pose + Root Position Head

Instead of directly predicting all absolute 3D joints, split the output into:

```text
root_relative_pose = model_pose_head(feature)
root_position      = model_root_head(feature)
final_pose         = root_relative_pose + root_position
```

Possible loss:

```text
L = L_abs_pose + lambda_rel * L_root_relative + lambda_root * L_root
```

Why this may help:

- PA-MPJPE is relatively stable, so pose structure is not the main failure.
- MPJPE is sensitive to absolute translation and scale.
- A separate root branch may reduce interference between pose shape and global
  position learning.

### 2. Subject-Robust Training

Treat each training subject as a group and optimize for robust performance.

Simple version:

```text
compute per-subject training loss
upweight high-loss subjects
```

More formal version:

```text
Group DRO / worst-group reweighting
```

Why this may help:

- The strict split evaluates unseen subjects.
- Current validation-to-test gap suggests subject shift.
- Optimizing only average training loss may overfit easier subjects.

### 3. Domain-Invariant WiFi Feature Alignment

Add a lightweight domain alignment objective across training subjects.

Candidate losses:

```text
Deep CORAL
MMD
subject adversarial loss with gradient reversal
```

Example:

```text
L_total = L_pose + lambda_cmc * L_cmc + lambda_domain * L_CORAL
```

Why this may help:

- CSI features are sensitive to subject, position, environment, and body shape.
- A subject-invariant representation may improve strict test MPJPE.

### 4. Temporal Positive CMC

Current CMC likely uses same-frame WiFi/RGB feature pairs. A stronger version can
use nearby frames or clips as positives.

Possible positives:

```text
same frame
nearby frames in the same action clip
same action segment
```

Possible negatives:

```text
different action
far frames
different subject
```

Why this may help:

- WiFi single-frame features can be noisy.
- Temporal positives encourage stable action/pose representations.

### 5. Skeleton and Temporal Regularization

Auxiliary losses:

```text
bone length consistency
left-right bone symmetry
temporal velocity smoothness
temporal acceleration smoothness
```

Why this may help:

- These losses improve pose plausibility.
- They may not solve MPJPE alone, but can stabilize training and reduce extreme
  predictions.

## Experiment Priority

Recommended order:

1. Search the strict WiFi baseline upper bound.
2. Search mixup alpha, learning rate, dropout, and batch size for baseline.
3. Keep root/relative auxiliary losses disabled unless new evidence appears.
4. Use conservative CMC scheduling after the baseline improves.
5. Tune CMC as a weak structural regularizer, not as a strong encoder loss.
6. If strict MPJPE remains above 240 mm, revisit split composition or train
   data size before adding more losses.
7. Consider subject-domain regularization only after baseline hyperparameters
   have been revalidated.

Useful local script:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode both --seeds 0
```

For a smaller first pass:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode baseline --seeds 0 --lrs 7e-5 1.03e-4 1.5e-4 --mixup_alphas 0.0 0.2 0.4 --dropouts 0.15 0.2
```

Follow-up narrow baseline search after finding `234.48 mm`:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode baseline --seeds 0 --lrs 5e-5 6e-5 7e-5 8e-5 9e-5 --mixup_alphas 0.3 0.4 0.5 0.6 --dropouts 0.1 0.15 0.2 --batch_sizes 64
```

After the best single-seed candidate is found, rerun it with multiple seeds:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode baseline --seeds 0 1 2 --lrs BEST_LR --mixup_alphas BEST_ALPHA --dropouts BEST_DROPOUT --batch_sizes 64
```

Reproducible stability search after the `234.48 mm` run was lost:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode baseline --seeds 0 --lrs 7e-5 --mixup_alphas 0.0 0.1 0.2 --dropouts 0.15 --transformer_layers 2 --dim_feedforwards 1024 --pose_head_hiddens 512 --batch_sizes 64 --aug_noises 0.0 0.02 --aug_freq_masks 0.0 0.05 --aug_time_masks 0.0 0.05 --lambda_rel_poses 0.0 0.2 --subject_robust_weights 0.0 0.1
```

Smaller first pass:

```bash
python run_strict_mpjpe_grid_search.py data_base config.yaml --mode baseline --seeds 0 --lrs 7e-5 --mixup_alphas 0.0 0.1 0.2 --dropouts 0.15 --transformer_layers 2 --dim_feedforwards 1024 --pose_head_hiddens 512 --batch_sizes 64 --aug_noises 0.02 --aug_freq_masks 0.05 --aug_time_masks 0.05 --lambda_rel_poses 0.0 0.2 --subject_robust_weights 0.0 0.1
```

For any finished run, diagnose the final-test gap:

```bash
python evaluate_final_test_breakdown_mpjpe.py data_base config.yaml --run_dir RUN_DIR --model_type baseline --num_workers 0
```

## Paper-Framing Notes

The current method can still be described around three main ideas:

1. Privileged RGB information during training.
2. Cross-modal contrastive matching between WiFi and RGB teacher features.
3. Cross-domain / cross-subject evaluation.

Mixup is better placed in the experimental setting or ablation section, not as a
main model contribution.

If root-relative + root-position is added later, it can be framed as an
MPJPE-oriented refinement for absolute 3D coordinate prediction.

## Related Work And Keywords

Useful papers / directions to cite or inspect:

- MM-Fi dataset: multimodal non-intrusive 4D human dataset.
- MetaFi: WiFi-enabled transformer-based human pose estimation.
- AdaPose: cross-site WiFi pose estimation and domain shift.
- Person-in-WiFi: early WiFi-based 3D human pose estimation.
- Wi-Mose: WiFi CSI amplitude/phase for pose and position information.
- GraphPose-Fi: graph-based WiFi 3D human pose estimation.
- Mixup: data augmentation and regularization.
- Deep CORAL: domain alignment by correlation matching.
- Group DRO: worst-group robust optimization.
- Learning using privileged information / teacher-student distillation.

Useful search keywords:

```text
WiFi CSI 3D human pose estimation cross subject generalization
WiFi human pose estimation domain adaptation CSI
WiFi CSI pose estimation contrastive learning
MPJPE root relative 3D human pose estimation
Deep CORAL domain adaptation loss
Group DRO worst group generalization
Learning using privileged information teacher student distillation
```
