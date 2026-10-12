# Lab Notebook

## 2026-10-01 Step 1.3 data acquisition
- figshare v8, 3064 files, licence CC BY 4.0
- sartaj 3, 3264 files, licence MIT
- br35h 12, 3861 files, licence Data files © Original Authors
- Figshare patients 233, exact duplicate files across sources 1274

## 2026-10-01 Step 1.4 deduplication and splits
- Clean images 3809, splits {'test': 2342, 'train': 1102, 'val': 365}
- Removed {'removed_duplicate': 3930, 'kept': 3809, 'removed_label_conflict': 60, 'removed_conflict_vs_figshare': 29}

## 2026-10-01 Step 1.4 deduplication and splits (revision 2)
- Clean images 3845, splits {'train': 2663, 'test': 788, 'val': 394}
- Removed {'removed_duplicate': 3934, 'kept': 3845, 'removed_label_conflict': 26, 'removed_conflict_vs_figshare': 23}

## 2026-10-01 Step 1.4 deduplication and splits (revision 3)
- Clean images 3845, splits {'train': 2610, 'test': 813, 'val': 422}
- Removed {'removed_duplicate': 3934, 'kept': 3845, 'removed_label_conflict': 26, 'removed_conflict_vs_figshare': 23}

## 2026-10-06 Step 1.4 closed, Step 2.1 code committed
- curated dataset verified, 3845 images, 0 array hash mismatches, 3010 non-empty Figshare masks
- pipeline code committed, 19 tests pass on CPU (torch 2.14.1, opacus 1.6.0, dp-accounting 0.6.0)
- synthetic learnability check after 8 rounds, non-DP test accuracy 0.51, DP at epsilon 8 test accuracy 0.44, chance 0.25
- next, run notebooks/kaggle/04-step2-pilot on T4 x2 with tag v0.2-rc1

## 2026-10-06 Step 2.3 pilot results (code v0.2-rc1, Kaggle T4 x2, about 0.5 GPU hours)
- tests 19 of 19 pass on Kaggle (torch 2.11.0), smoke run fine, 28 of 28 pilot runs finished and pushed
- timing per 60-round run, small CNN 64 s (no DP) and 108 s (DP) on dir0.5_k10, 80 s and 92 s on natural3, ResNet-18 GN 178 s and 440 s
- DP calibration, sigma 1.756 for epsilon 8 at q 0.1 and 600 steps
- utility too low, best small CNN without DP test accuracy 0.61 (dir0.5_k10, lr 0.05), train accuracy 0.52 to 0.69, meningioma sensitivity near 0
- natural3 without DP, no-tumour sensitivity 0 for every model, consistent with FedAvg drift when 78 percent of weight sits on Figshare, which has no normal scans
- with DP at epsilon 8, best validation round often 7 to 16 of 60, validation loss rises afterwards
- decision, add Step 2.4 diagnostics before Week 3 (D27)

## 2026-10-06 Step 2.4 diagnostics (code v0.2-rc2, about 1.8 GPU hours)
- first background attempt failed on a Kaggle dataset mount error, rerun succeeded, 23 of 28 runs finished
- all five non-DP resnet18_frozenbn runs failed, four NaN from round 1, one collapsed to a single class (balanced accuracy 0.25), DP runs of the same model trained normally
- centralised DP epsilon 8, ResNet frozen-BN lr 0.01, balanced accuracy 0.851, test accuracy 0.862, sensitivities glioma 0.92, meningioma 0.64, no tumour 0.86, pituitary 0.97, best round 100
- centralised small CNN, no DP best balanced accuracy 0.73, DP best 0.57, meningioma sensitivity at most 0.22
- federated DP epsilon 8, ResNet lr 0.01, natural3 balanced accuracy 0.73 (best round 17), dir0.5_k10 0.62 (best round 14), lr 0.05 unstable
- federated small CNN without DP, natural3 lr 0.05 keeps all classes (no-tumour sensitivity 0.50, meningioma 0.37), so the earlier no-tumour collapse was largely undertraining
- decisions D28 and D29

## 2026-10-06 Step 2.4 tuning (code v0.2-rc3, about 1.4 GPU hours, 17 of 17 runs, none diverged)
- central lr 0.02 balanced accuracy, no DP 0.918, clip-only 0.893, DP epsilon 8 0.849, meningioma sensitivity 0.90, 0.80, 0.62
- central lr 0.01, no DP 0.908, clip-only 0.861, DP 0.825, lr 0.003 clearly worse under clipping (0.73)
- federated no DP, natural3 0.848 (lr 0.01), dir0.5_k10 0.888 (lr 0.01)
- federated DP epsilon 8, natural3 0.806 (lr 0.01, smooth curve), dir0.5_k10 0.714 (lr 0.003) and 0.613 (lr 0.01, best round 14, validation loss rising while balanced accuracy climbs)
- exit criteria met, central no DP balanced accuracy 0.918 at least 0.85, federated DP natural3 keeps every class sensitivity at least 0.60
- settings frozen (D30), Week 3 launcher 07-week3-core prepared

## 2026-10-07 Step 3.1 Week 3 core (code v0.2-pipeline-verified, 6.8 GPU hours, 83 of 83 runs, none diverged)
- central, seeds 1 to 5, balanced accuracy with t-based 95% interval: no DP 0.919 (0.907 to 0.931), clip-only 0.871 (0.850 to 0.891), epsilon 8 0.815 (0.779 to 0.851), epsilon 3 0.686 (0.616 to 0.756), epsilon 1 0.331 (0.269 to 0.393)
- clipping alone costs 4.8 points and noise at epsilon 8 a further 5.6, the single-seed Step 2.4 estimates (0.893, 0.849) were optimistic
- meningioma sensitivity falls 0.861, 0.711, 0.520, 0.179, 0.086 across the same regimes, tumour missed as normal stays at or below 0.042
- epsilon 1 is near collapse, best round 2, macro AUC 0.663
- local baselines (seeds 1 to 3): Figshare client has no no-tumour images (balanced accuracy capped at 0.75), no DP 0.625, epsilon 8 0.464; SARTAJ client no DP 0.543, epsilon 8 0.245; Br35H client holds one class, so its local model is degenerate by construction (0.250)
- federated lr selection at seed 0: every non-DP configuration picks 0.01, every DP configuration picks 0.003 except natural3 at epsilon 8 (0.01)
- seed 0 is selection only, but the pattern is clear: without DP natural3 is the worst partition (0.848 against 0.89 to 0.90 for iid and dir5.0), at epsilon 8 it is the best (0.806 against 0.66 to 0.72), consistent with H5
- bug caught before any GPU time: expand() shared nested preset dicts, so a generated seeds grid collapsed every configuration to seed 3; the duplicate check in run_grid refused it; fixed with deep copies, test added, all 83 Week 3 run ids unchanged
- tables results/tables/week03_baselines_summary.{csv,md} and week03_lr_selection.csv, grids week03_fed_seeds_a (seeds 1 to 3, 60 runs) and week03_fed_seeds_b (seeds 4 and 5, 40 runs), decision D35

## 2026-10-07 Step 3.1 federated seeds (08-fed-seeds-a, code v0.3-unit-dp, 6.2 GPU hours, 60 of 60 runs; seeds 4 and 5 pushed early by 09)
- all 100 federated seed runs (20 configurations x seeds 1 to 5) are in, none diverged, table results/tables/week03_fed_summary_interim.{csv,md}
- balanced accuracy, mean over 5 seeds: no DP iid 0.904, dir5.0 0.899, dir0.5 0.878, natural3 0.875, dir0.1 0.843; epsilon 8 natural3 0.808, iid, dir5.0 and dir0.5 0.718 to 0.720, dir0.1 0.677; epsilon 3 natural3 0.714, others 0.623 to 0.639; epsilon 1 natural3 0.537, others 0.322 to 0.348
- H5 holds with 5 seeds. Without DP label skew matters (dir0.1 3.2 points below natural3, natural3 2.9 below iid). Under DP natural3 beats iid by 8.8, 7.5 and 18.9 points at epsilon 8, 3 and 1 (Welch p at most 0.0022), while iid, dir5.0 and dir0.5 are indistinguishable at epsilon 8
- the epsilon 3 and epsilon 1 comparisons use the same lr (0.003) for every partition, so the natural3 advantage is not an lr artefact (natural3 at epsilon 8 uses 0.01)
- mechanism: at a fixed total of N images, FedAvg's aggregated noise standard deviation is sigma C sqrt(K) / (q N), so 3 clients carry about 1.8 times less noise than 10. Client size and client count are the same lever at fixed N; Step 3.4 (dir0.5 with K = 3, 5, 10, 15) isolates it
- federated natural3 at epsilon 8 (0.808) matches central epsilon 8 (0.815), so federation itself costs almost nothing under DP, whereas without DP it costs 4.4 points (0.875 against 0.919)
- meningioma sensitivity at epsilon 8 is 0.59 on natural3 and 0.18 to 0.24 elsewhere

## 2026-10-07 Step 3.1b patient-level DP, first results (09-fed-seeds-b-unit, code v0.3-unit-dp, 6.65 GPU hours, 64 of 64 runs, none diverged)
- patient-level central DP collapses: epsilon 8 balanced accuracy 0.353 (0.293 to 0.413), epsilon 3 0.253 (0.245 to 0.261), against 0.815 and 0.686 at record level with the same sigma, q and steps
- the collapsed models predict no tumour: no-tumour sensitivity 0.89 (epsilon 8) and 0.99 (epsilon 3), tumours missed as normal 0.54 and 0.98, against 0.025 at record level
- two causes. (1) Noise: 634 units against 2,610 images, so the noise in the averaged gradient, sigma C / (q U), is 4.1 times larger, which sits between record-level epsilon 3 and epsilon 1. (2) Weighting: each patient gets one vote, and no-tumour units are 58% of units but 17% of images (glioma 14% of units, 38% of images), so the effective class prior shifts towards no tumour
- federated patient-level lr selection at seed 0 is at or near chance in all 8 runs (balanced accuracy 0.23 to 0.47)
- udir0.5_k10 record-level selection at seed 0: no DP 0.891 (lr 0.01), epsilon 8 0.729 (lr 0.003), epsilon 3 0.661 (lr 0.003), close to dir0.5_k10, so keeping patients whole does not change the record-level picture at seed 0
- natural3 patient-level runs report unit_epsilon_worst 11.98 at a target of 8, because 11 look-alike units cross two clients
- decision D36, launcher 10 (45 runs)

## 2026-10-08 Step 3.1b extension and udir seeds (10-unit-udir, code v0.3.1-unit-ext, 4.97 GPU hours, 45 of 45 runs, none diverged)
- patient-level clip-only (zero noise) reaches 0.884 (0.866 to 0.903), no worse than record-level clip-only (0.871), so one vote per patient and per-patient clipping cost nothing; the D36 class-prior explanation is refuted, the collapse is the noise
- patient-level central recovery curve: epsilon 3 0.253, epsilon 8 0.353, epsilon 16 0.584, epsilon 32 0.745, epsilon 64 0.836 (0.800 to 0.873); patient-level epsilon 64 matches record-level epsilon 8 (0.815)
- federated patient-level epsilon 8: natural3 0.403 (worst-unit epsilon 11.98), udir0.5_k10 0.264 (chance, tumours missed 0.50)
- record-level udir0.5_k10 (patients kept whole): no DP 0.904, epsilon 8 0.715, epsilon 3 0.640, statistically indistinguishable from dir0.5_k10 (0.878, 0.718, 0.626), so image-level partitioning did not distort the record-level results
- Step 3.2 completed on CPU: FedAvg at epsilon 3 and 1 was already in the Step 3.1 grid, figure figures/fig_privacy_utility.{png,pdf} (scripts/plot_privacy_utility.py), Gate A recorded in D37
- launcher 11 (client-count sweep, 45 runs) prepared for tag v0.3.2-client-count

## 2026-10-09 Step 3.3 preparation
- SCAFFOLD normaliser corrected for momentum (D38); old formula gave c_i = [-3.10, 1.03, 1.03, 1.03] against the true constant gradient [-0.75, 0.25, 0.25, 0.25] in the new test, the corrected one recovers it
- select_lr.py generalised with --param (used for fl.mu)
- grids week06_select_fedprox (30), week06_scaffold_seeds_a (45), week06_scaffold_seeds_b (30); launcher 12 for tag v0.4-step33; 34 tests pass

## 2026-10-10 Step 3.4 client-count sweep (11-client-count, code v0.3.2-client-count, 4.16 GPU hours, 45 of 45 runs, none diverged)
- dir0.5 with K = 3, 5, 10, 15 (K = 10 from Step 3.1), balanced accuracy over 5 seeds: no DP 0.868, 0.906, 0.878, 0.903 (no trend, slope per ln K +0.012, p 0.08); epsilon 8 0.717, 0.735, 0.718, 0.696 (slope -0.014, p 0.40); epsilon 3 0.719, 0.708, 0.626, 0.558 (slope -0.10 per ln K, p < 0.0001)
- meningioma sensitivity at epsilon 3 falls 0.315, 0.171, 0.067, 0.030 as K grows
- with size-weighted FedAvg the aggregated noise std is sigma C sqrt(K) / (q N) for any client sizes; plotted against it, the epsilon 8 and epsilon 3 points form one curve that is flat to about 0.04 and falls after, figures/fig_h5_noise.{png,pdf}, table results/tables/week05_h5_mechanism.csv
- at epsilon 3 natural3 (0.714) equals dir0.5_k3 (0.719) under the same lr, so client count explains natural3's advantage there; at epsilon 8 natural3 (0.808, lr 0.01) is far above dir0.5_k3 (0.717, lr 0.003), which is either the lr or the source structure, hence the lr 0.01 check in launcher 12 (D39)

## 2026-10-11 Step 3.4 lr check and Step 3.3 first results (12-fedprox-select-scaffold-a, code v0.4-step33, 9.48 GPU hours, 95 of 95 runs, none diverged)
- epsilon 8 with lr 0.01: K = 3, 5, 10, 15 give 0.820, 0.773, 0.662, 0.564 (slope -0.158 per ln K, p 4e-7), against 0.717, 0.735, 0.718, 0.696 with lr 0.003. H5 holds at epsilon 8 once lr is not handicapped; the best lr depends on K (lr 0.01 for K up to 5, 0.003 for K 10 and 15), so large clients tolerate larger steps under DP. dir0.5_k3 at lr 0.01 (0.820) matches natural3 (0.815), so natural3's epsilon 8 lead is client count plus lr, not source structure
- FedProx mu 0.01 and 0.1 at seed 0 are indistinguishable from FedAvg in all 15 configurations (max |delta best val loss| 0.010, max |delta balanced accuracy| 0.028); with one local epoch at lr 0.003 to 0.01 the drift w - w_global is too small for the proximal term to act
- SCAFFOLD seeds 1 to 3 (uniform control-variate averaging) against FedAvg: neutral on iid, dir5.0 and dir0.5 without DP, but harmful where client sizes are unequal and noise is high, natural3 -0.22 and -0.36 at epsilon 8 and 3, dir0.1 -0.08, -0.25, -0.34 at no DP, epsilon 8, 3. Harm follows the client-size spread (coefficient of variation natural3 1.16, dir0.1 1.34, dir0.5 0.58, dir5.0 0.20, iid 0)
- diagnosis: the server model is size-weighted but c was a uniform mean of c_j, so every client was pulled towards the unweighted mean gradient, giving the 230-image single-class Br35H client a third of the weight in natural3 and amplifying the noise of tiny clients under DP; fixed with size-weighted control variates (D40), 35 tests pass

## 2026-10-12 Step 3.3 closed (13-scaffold-size, code v0.4.1-scaffold-size, 10.02 GPU hours, 105 of 105 runs, none diverged)
- FedProx mu 1 and 10 at seed 0, change in best validation loss against FedAvg: mu 1 within -0.013 to +0.038, mu 10 worse without DP and at epsilon 8 (+0.02 to +0.19) and at most 0.064 better at epsilon 3, always inside one FedAvg seed standard deviation (0.016 to 0.077); by the D41 rule no FedProx seed runs follow, table results/tables/week06_fedprox_vs_fedavg_seed0.csv
- size-weighted SCAFFOLD against FedAvg, paired over seeds 1 to 5, Holm over 15 configurations: no difference survives correction (smallest Holm p 0.064, natural3 no DP +0.025); largest raw changes natural3 epsilon 8 +0.017, dir0.1 no DP -0.028, dir0.1 epsilon 8 +0.028 (p 0.11); table results/tables/week06_scaffold_vs_fedavg.csv
- uniform-weighted SCAFFOLD (the D40 ablation) against the same configurations: natural3 epsilon 8 0.590 and epsilon 3 0.358, dir0.1 epsilon 3 0.268, against 0.825, 0.713, 0.633 with size weighting
- RQ2 answer: with one local epoch, drift correction neither helps nor hurts FedAvg under DP once it is implemented consistently with the aggregation weights; implemented inconsistently, it fails badly exactly where clients are unequal and noise is high
- Phase 3 complete, 616 GPU runs so far, about 52 GPU hours in total

## 2026-10-12 Phase 4 code and source controls (code v0.5-audit, CPU only)
- new modules src/xai/maps.py (Grad-CAM, Grad-CAM++, Integrated Gradients with a black baseline, 32 steps), src/xai/metrics.py (pointing game with 3-pixel tolerance, top 15% IoU, energy in mask, deletion and insertion AUC over 20 steps, Spearman rows, patient bootstrap), src/xai/audit.py, src/attacks/mia.py, src/attacks/probe.py; scripts run_audit.py, summarise_audit.py, source_controls.py; make_launcher.py --kind audit builds 14-audit; 40 tests pass
- launcher 14 audits 165 checkpoints (MIA and probe) and explains 115 of them, using the outputs of 07 to 10 as inputs, no training
- source controls on saved test predictions, results/tables/week07_source_controls.{csv,md}: SARTAJ-only 4-class balanced accuracy trails the pooled figure by 0.25 to 0.29 in every non-collapsed regime (central no DP 0.663 against 0.919, epsilon 8 0.528 against 0.815); no-tumour specificity is near 1 on Br35H and lower on SARTAJ, and DP widens the gap (central epsilon 8 0.702 against 0.976); FedAvg natural3 no DP has SARTAJ specificity 0.678, the Br35H client being single-source no-tumour
- caveat: SARTAJ test holds 102 images (13 G, 24 M, 59 N, 6 P), so per-source intervals are wide

## 2026-10-12 Phase 4 audit (14-audit, code v0.5-audit, about 1.7 GPU hours, 165 of 165 models, none failed)
- 115 models explained (about 50 s each), 11 randomisation checks, 165 MIA and probe; table results/tables/week07_audit_summary.{csv,md} plus _per_run.csv
- MIA image AUC no DP 0.53 to 0.54, patient mean AUC 0.68 (central, FedAvg natural3), 0.61 to 0.64 (dir0.5, dir5.0, iid); every clipped or DP model 0.46 to 0.54
- probe tumour Figshare vs SARTAJ 0.91 no DP, 0.85 at epsilon 1; no-tumour SARTAJ vs Br35H 0.82 to 0.77
- Grad-CAM pointing 0.197 to 0.089 and IoU 0.087 to 0.032 from no DP to epsilon 1 (central), explained by accuracy (r 0.85, sigma p 0.51); IG fails the randomisation check (Spearman 0.68 to 0.71); D44
- bug: the summary cell ran git pull against the token-free remote and failed; summary produced here instead, cell fixed in make_launcher.py

## 2026-10-12 Launcher 15 prepared (code v0.6-lira)
- 64 LiRA shadows (natural3 at no DP, epsilon 8, 3, 1, 16 each), weight-noise selection (5) then seeds (5 per sigma) and shadows (16 per sigma), audit extensions on all 165 earlier models; about 5.5 hours; D45
- 15 first attempt: tests passed, 5 weight-noise selection runs finished (sigma 0.5 diverged, 0.05 and 0.01 poor, 0.003 and 0.001 near no-DP), then the grid generator crashed on a null val_loss; fixed in v0.6.1-lira, chosen sigmas 0.5 and 0.003 (D46)

## 2026-10-12 Launcher 15 (code v0.6.1-lira, about 4.7 GPU hours of training plus audit)
- weight noise 0.5 diverged in all 21 runs, balanced accuracy 0.259; weight noise 0.003 0.843 over 5 seeds
- probe baselines pretrained 0.871 / 0.783, random 0.846 / 0.776 (tumour Figshare vs SARTAJ / no-tumour SARTAJ vs Br35H)
- Grad-CAM rim energy 0.17 to 0.19 against chance 0.33, outside-brain energy 0.26 to 0.36; layer-3 Grad-CAM no better
- logits.npz and cams8.npz not pushed (.gitignore *.npz); summary cell failed on inf logits; fixes and launcher 16 in v0.6.2-audit (D47)
- 16 first attempt stopped at the audit cell: /kaggle/working/logs did not exist (created only in launcher 15's selection cell); fixed in the notebook generator, code tag v0.6.2-audit unchanged

## 2026-10-12 Launcher 16 (code v0.6.2-audit, about 0.3 GPU hours, 271 logits files, 115 Grad-CAM map files)
- LiRA online, natural3: no DP AUC 0.662, TPR@1% 0.112, TPR@0.1% 0.033, patient AUC 0.805; epsilon 8, 3, 1 TPR@1% 0.015, 0.017, 0.009; weight noise 0.003 AUC 0.589, TPR@1% 0.047, patient AUC 0.745
- within-stratum loss AUC under DP still below 0.5 (natural3 epsilon 8 0.459)
- consistency IoU central 0.700 / 0.663 / 0.485 / 0.309 / 0.235 (no DP, clip, eps 8, 3, 1); r 0.79 with accuracy
- outside-brain chance 0.38 to 0.46, Grad-CAM outside energy at or below it; Steps 4.1 and 4.2 closed (D48)
- 2026-10-12 NCC cross-check prepared (D49): notebook 17, CPU only, code v0.7-ncc
- 2026-10-12 NCC check: 66 contaminated test images, mostly mirrored Br35H no-tumour; author chose the clean test set as primary, no retraining (D50, D51)
- 2026-10-12 paper figures fig1 to fig4 on the clean test set (scripts/make_figures.py, figures/CAPTIONS.md); fix: image ids with spaces were split when reading contaminated_test_ids.txt (63 instead of 66 excluded in the first clean LiRA table), table regenerated
- 2026-10-12 notebook 18 prepared so that tables and figures are regenerated on Kaggle (CPU) from GitHub files; local dry run identical to committed figure data
- 18 first attempt: 12 tests failed, opacus missing because the figures setup cell skipped pip install -r requirements.txt; fixed in the generator, code tag v0.8-figures unchanged

## 2026-10-12 Results freeze (notebook 18, code v0.8-figures, CPU)
- 52 tests pass on Kaggle; tables and figures regenerated, figure data identical to committed (max difference 0)
- results/FROZEN_MANIFEST.{json,md}: 21 grids, 608 runs all finished, frozen files hashed; Step 4.3 closed (D52)
- 2026-10-12 D53: FedAvg and SCAFFOLD seeds were pooled in week07_ncc_sensitivity and in Figures 1 and 4; fixed, freeze withdrawn pending a Kaggle rerun of notebook 18 (v0.8.1-figures)
- 2026-10-12 Step 5.1 started: full first draft (Intelligent Oncology, KeAi guide for authors) written as a Docs artifact from frozen and D53-corrected numbers; 12 references marked for verification
- 2026-10-13 notebook 18 on v0.8.1-figures: all four figure data files identical to committed; manifest rebuilt, D54
- 2026-10-13 Step 5.1 closed, 5.2 in progress: manuscript in cas-sc LaTeX with Word export and 11-page supplement; 32 references verified, 3 rounding errors in Table 2 fixed (D55)
