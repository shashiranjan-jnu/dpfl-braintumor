# Changelog

Newest entry first.

## 2026-10-13 (official template)

- Manuscript moved to the journal's official elsarticle class (`paper/elsarticle/`), CAS stand-in removed, tables fitted to the text block, Latin Modern fonts; `make_figures.py` embeds TrueType fonts (D56).

## 2026-10-13 (v0.9.2-manuscript, proposed)

- `paper/`: manuscript in Elsevier CAS LaTeX (`main.tex`), AMA references verified (`references.py`), build script producing the submission folder with preview PDF, Word file, numbered figures, highlights and supplement PDF (D55).
- `scripts/make_supplement.py`: supplementary Tables S1 to S18 from frozen files.

## 2026-10-13 (v0.9.1-results-frozen)

- Kaggle rerun under v0.8.1-figures matches the corrected figures exactly; freeze manifest rebuilt (D54).

## 2026-10-12 (v0.8.1-figures)

- Fix: clean-set summary pooled FedAvg with size-weighted SCAFFOLD; `summarise.py` now groups by algorithm when several are present, figures use FedAvg only. Test added. Notebook 18 points at `v0.8.1-figures`. Freeze v0.9 withdrawn, re-freeze as v0.9.1 after the Kaggle run (D53).

## 2026-10-12 (v0.9-results-frozen)

- `scripts/freeze.py` writes and checks `results/FROZEN_MANIFEST.{json,md}`. Notebook 18 reproduced figures exactly on Kaggle. D52, Step 4.3 closed.

## 2026-10-12 (v0.8-figures)

- `make_figures.py --results --out`; `make_launcher.py --kind figures` builds notebook `18-figures` (CPU), which regenerates the clean-set tables and figures 1 to 4 on Kaggle from the files on GitHub, compares them with the committed figure data and pushes them. Local dry run reproduces the committed figure data exactly.

## 2026-10-12 (paper figures)

- `scripts/make_figures.py`: fig1 privacy-utility, fig2 LiRA ROC, fig3 explanation consistency, fig4 H5, all on the clean test set, with `figures/figure_data/*.csv` and draft captions. `scripts/ncc_sensitivity.py` and `mia_tables.py --exclude`. Fix: id list read line by line (ids contain spaces).

## 2026-10-12 (v0.7-ncc)

- `scripts/ncc_check.py` (curated all-pairs NCC with flips, removed-duplicate re-check from raw sources), `tests/test_ncc.py`, `make_launcher.py --kind ncc`, notebook `17-ncc-dedup-check` (CPU). 51 tests pass. D49.

## 2026-10-12 (Steps 4.1 and 4.2 closed)

- 16 finished: logits for 271 models and Grad-CAM maps for 115 now on GitHub. Tables `week07_lira`, `week07_mia_stratified`, `week07_xai_border` (with consistency), `week07_wn_seeds_summary`. D48.

## 2026-10-12 (v0.6.2-audit)

- 15 finished training (106 runs, about 5.4 hours) but its .npz audit files were ignored by git. Fix: `.gitignore` re-includes `results/audit/**/*.npz`, parts with missing files are redone, logits float32, non-finite models skipped, weight-noise regimes labelled, border analysis reports the outside-brain chance level. Launcher `16-audit-logits`. Tables `week07_wn_seeds_summary`, `week07_xai_border` (partial). D47.

## 2026-10-12 (v0.6.1-lira)

- Fix: best validation loss skips rounds whose loss was non-finite (stored as null) in `aggregate.py`, `select_lr.py`, `make_shadow_grid.py`. Test added, 48 pass. Launcher 15 points at `v0.6.1-lira`. D46.

## 2026-10-12 (launcher 15 prepared)

- New optional keys `data.shadow` (LiRA shadow subsets, whole units of train plus test) and `fl.weight_noise` (Gupta-style comparator, refused with DP). Existing run ids unchanged.
- `run_audit.py --parts`, `--probe-baselines`, several `--ckpt-root`; new parts `logits` and `border`. Layer-3 Grad-CAM, brain-rim regions, consistency IoU.
- `scripts/make_shadow_grid.py`, `scripts/mia_tables.py` (stratified loss attack, online and offline LiRA), `scripts/xai_tables.py`. Grids `week07_wn_select`, `week07_lira_shadows`. Launcher `15-lira-wn` for tag `v0.6-lira`. 47 tests pass. D45.

## 2026-10-12 (launcher 14 results)

- 14 finished, 165 models audited. Table `week07_audit_summary.{csv,md}` and per-run CSV. D44, Gate B passes with the utility caveat.
- Fix: audit summary cell no longer pulls from the token-free remote. `summarise_audit.py` leaves cells blank for parts not run.

## 2026-10-12 (Phase 4 code)

- New: `src/xai/{maps,metrics,audit}.py`, `src/attacks/{mia,probe}.py`, `scripts/run_audit.py`, `scripts/summarise_audit.py`, `scripts/source_controls.py`, `tests/test_audit.py` (40 tests pass).
- `make_launcher.py --kind audit` builds `14-audit` for tag `v0.5-audit`; checkpoints come from the outputs of launchers 07 to 10 (D43).
- Synthetic data gives Figshare tumour images a mask. `summarise.py` takes a metric map and orders patient epsilon 16 to 64.
- Table `week07_source_controls.{csv,md}`.

## 2026-10-12 (Step 3.3 closed)

- 13 finished, 105 runs, 10.02 GPU hours. Tables `week06_scaffold_vs_fedavg.csv` (Holm-corrected) and `week06_fedprox_vs_fedavg_seed0.csv`. D42, Phase 3 complete.

## 2026-10-11 (Step 3.3 first results)

- 12 finished, 95 runs, 9.48 GPU hours.
- Fix: SCAFFOLD control variates can be size-weighted (`fl.scaffold_c: size`, D40); default `uniform` keeps old run ids. Test added, 35 pass.
- Grid `week06_scaffold_size` (75 runs), launcher `13-scaffold-size` for tag `v0.4.1-scaffold-size`. Proposed grid `week06_select_fedprox_ext` (mu 1 and 10).

## 2026-10-10 (Step 3.4 results)

- 11 finished, 45 runs, 4.16 GPU hours. Tables `week05_client_count_all.csv`, `week05_h5_mechanism.csv`; `scripts/plot_h5.py`, `figures/fig_h5_noise.{png,pdf}`.
- Grid `week05_client_count_lr` (20 runs) added to launcher 12, which now holds 95 runs, about 9.4 GPU hours (D39).

## 2026-10-09 (Step 3.3 preparation)

- Fix: SCAFFOLD control variates now use the momentum-correct normaliser lr S(K, beta) and count applied steps (D38). Two tests added, 34 pass.
- `select_lr.py --param` selects any dotted hyperparameter (fl.mu).
- Grids `week06_select_fedprox`, `week06_scaffold_seeds_a`, `week06_scaffold_seeds_b`; launcher `12-fedprox-select-scaffold-a` for tag `v0.4-step33`.

## 2026-10-08 (patient-level extension, Step 3.2)

- 10 finished, 45 runs, 4.97 GPU hours. Tables `week04_privacy_unit_central.{csv,md}`, `week04_fed_all.{csv,md}`.
- `scripts/plot_privacy_utility.py` and `figures/fig_privacy_utility.{png,pdf}`.
- Grid `week05_client_count` (45 runs) and launcher `11-client-count` for tag `v0.3.2-client-count` (D37).

## 2026-10-07 (patient-level results)

- 09 finished, 64 runs, 6.65 GPU hours. `results/tables/week04_unit_central_summary.{csv,md}`, lr selection tables for udir and unit.
- Grids `week04_udir_seeds` (15), `week04_unit_fed` (10, epsilon 8 only), `week04_unit_central_ext` (20, patient clip-only and epsilon 16, 32, 64). Launcher `10-unit-udir` for tag `v0.3.1-unit-ext` (D36).

## 2026-10-07 (Week 3 results)

- Week 3 core finished, 83 runs, 6.8 GPU hours, tables pushed by the launcher.
- `scripts/summarise.py`: seed means with t-based 95% intervals (D34), `results/tables/week03_baselines_summary.{csv,md}`.
- lr selection done, `configs/grids/week03_fed_seeds_a.yaml` (seeds 1 to 3) and `_b.yaml` (seeds 4 and 5), table `results/tables/week03_lr_selection.csv`.
- Fix: `expand()` shared nested preset dicts, so axis values leaked between configurations; `select_lr.py` no longer copies the selection seed. Test added, 32 tests pass, Week 3 run ids unchanged.
- Launchers `08-fed-seeds-a` and `09-fed-seeds-b-unit` for tag `v0.3-unit-dp` (D35).

## 2026-10-06 (patient-level DP)

- Unit-level (patient-level) DP: `clipped_unit_grad_sum` in `src/privacy/dp.py`, unit sampling in `src/fl/federated.py`, `dp.unit: unit` in configs (D33).
- Patient-respecting Dirichlet partitions `udir<alpha>_k<K>` computed in `src/data/data.py`.
- Runs report `units_max_clients`, `unit_epsilon_worst` and `epsilon_if_k_runs` (D33, D34). Record-level run ids unchanged.
- Grids `week04_unit_central`, `week04_select_udir`, `week04_select_unit`. Unit-level lr is selected at unit level, `epsilon_if_k_runs` composes over clients and candidates, unit-level DP is refused on iid and dir partitions (after code review). 31 tests pass.
- Gupta 2026 and Kaissis 2021 read in full, `docs/related_work.csv` and `docs/gap_statement.md` updated, D34.

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
