# Results freeze manifest

Built from commit 41e2e39767. Test set: clean, 747 images (D51); full 813 in the supplement. Verify with `python scripts/freeze.py --check`.

| Group | Grid | Runs | Finished |
|---|---|---|---|
| utility, central record and patient level | week03_central | 25 | 25 |
| utility, central record and patient level | week04_unit_central | 10 | 10 |
| utility, central record and patient level | week04_unit_central_ext | 20 | 20 |
| utility, FedAvg partitions | week03_fed_seeds_a | 60 | 60 |
| utility, FedAvg partitions | week03_fed_seeds_b | 40 | 40 |
| utility, FedAvg partitions | week04_unit_fed | 10 | 10 |
| utility, FedAvg partitions | week04_udir_seeds | 15 | 15 |
| local-only baselines | week03_local | 18 | 18 |
| H5 client count | week05_client_count | 45 | 45 |
| H5 client count | week05_client_count_lr | 20 | 20 |
| FedProx and SCAFFOLD | week06_select_fedprox | 30 | 30 |
| FedProx and SCAFFOLD | week06_select_fedprox_ext | 30 | 30 |
| FedProx and SCAFFOLD | week06_scaffold_seeds_a | 45 | 45 |
| FedProx and SCAFFOLD | week06_scaffold_size | 75 | 75 |
| weight-noise comparator | week07_wn_select | 5 | 5 |
| weight-noise comparator | week07_wn_seeds | 10 | 10 |
| LiRA shadows | week07_lira_shadows | 64 | 64 |
| LiRA shadows | week07_wn_shadows | 32 | 32 |
| learning-rate selection (seed 0) | week03_select_fed | 40 | 40 |
| learning-rate selection (seed 0) | week04_select_udir | 6 | 6 |
| learning-rate selection (seed 0) | week04_select_unit | 8 | 8 |

Total 608 runs, 608 finished. 91 files hashed (tables, figures, figure data, NCC results, probe baselines, split and duplicate manifests).
