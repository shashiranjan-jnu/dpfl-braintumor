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
