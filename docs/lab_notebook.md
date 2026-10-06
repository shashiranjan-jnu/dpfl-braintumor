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
