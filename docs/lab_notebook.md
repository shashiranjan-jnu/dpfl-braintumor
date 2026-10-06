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
