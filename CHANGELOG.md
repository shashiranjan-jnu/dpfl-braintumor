# Changelog

Newest entry first.

## 2026-10-06 (late)
- Step 1.2 literature check, docs/related_work.csv (43 studies) and docs/gap_statement.md
- Decision D32, RQ3 narrowed, RQ4 patient-level, source-confound controls, NCC duplicate cross-check, unit-level DP proposed

## 2026-10-06 (night)
- Step 2.4 tuning ran (06-step2-tune, about 1.4 GPU hours), 17 of 17 runs, no divergence
- Settings frozen in configs/base.yaml (D30), findings recorded (D31)
- scripts/select_lr.py implements the seed-0 learning-rate selection protocol
- Week 3 grids week03_select_fed (40), week03_central (25), week03_local (18), launcher 07-week3-core
- Tests pin the small CNN offline, new selection test, 25 tests pass

## 2026-10-06 (evening)
- Step 2.4 diagnostics ran (05-step2-diag, about 1.8 GPU hours), 23 of 28 runs finished, 5 non-DP frozen-BN ResNet runs diverged
- Fixes: NaN guard with diverged flag, optional global-norm grad_clip for non-DP runs, cosine schedule, noise_multiplier 0 honoured (D28)
- Gate A: resnet18_frozenbn promoted to primary model (D29)
- New grids week02_tune_central (9 runs, includes clip-only ablation) and week02_tune_fed (8 runs), launcher 06-step2-tune
- 24 tests pass

## 2026-10-06 (afternoon)
- Step 2.3 pilot completed on Kaggle T4 x2, 28 runs, 0 failures, about 0.5 GPU hours, tables in results/tables/week02_*.csv
- Added resnet18_frozenbn (pretrained, BatchNorm frozen in eval mode, D25) and small_cnn width option
- data.version added to base config so run_ids encode the dataset (D26), pilot ids archived
- Grid files accept presets, aggregator reports clip fraction and per-class sensitivity, sklearn warnings muted
- scripts/make_launcher.py generates Kaggle launchers, notebooks/kaggle/05-step2-diag.ipynb for Step 2.4
- Two new tests for frozen BatchNorm (per-example independence, unchanged statistics, clipping bound), 21 tests pass

## 2026-10-06
- Step 1.4 passed. brain-mri-curated-v1 verified (curated_checks.json), data_flow.md committed
- Step 2.1 pipeline added. src/data, src/models (SmallCNN 97,908 params, ResNet-18 GN), src/privacy/dp.py (per-example clipping, Gaussian noise, accounting), src/fl/federated.py (FedAvg, FedProx, SCAFFOLD), src/eval/metrics.py, src/run.py
- scripts/run_grid.py shards grids across GPUs, skips finished runs, pushes each run with a file lock, token read from GIT_PUSH_URL
- scripts/aggregate.py builds summary tables, push_results.sh now honours GIT_PUSH_URL
- tests/ adds 19 tests covering clipping bound, sensitivity, zero-noise equivalence, micro-batching, noise scale, accountant agreement, calibration, partitions, determinism, and end-to-end runs
- configs/base.yaml plus Step 2.3 grids week02_lr, week02_pilot, week02_resnet, and notebooks/kaggle/04-step2-pilot.ipynb
- Decisions D19 to D24 recorded

## 2026-10-01
- Single-branch workflow adopted, all work and results committed directly to main, dev retired
- Step 1.3 manifests committed, namely raw_index.csv, sources.csv, acquisition_checks.json, exact_duplicate_summary.csv
- Br35H-Mask-RCNN labelled annotated_duplicate, and only Br35H no-tumour images retained (D1 to D4)
- SARTAJ redistribution corrected to yes under its MIT licence

## 2026-09-30
- Push script now uses rebase with autostash, because a crashed run can leave unstaged files that block a rebase
- Launchers use two working copies, code checked out at a tag and results committed on dev, because a tag checkout is detached and cannot push
- Kaggle secret label is kaggle_dpfl-braintumor instead of GH_TOKEN
- .gitignore pattern changed from data/ to /data/, because the unanchored pattern silently ignored src/data
- README expanded with repository layout and data policy
- Scaffold pushed from Kaggle notebook 00-scaffold-repo
- Roadmap created and committed as docs/ROADMAP.md
