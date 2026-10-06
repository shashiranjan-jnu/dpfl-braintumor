# Design Decisions
## 2026-09-30 Step 1.3 source composition

D1. Br35H-Mask-RCNN (801 images) duplicates Br35H yes exactly and is excluded.
D2. Br35H yes (1,500) lacks tumour type and is excluded from the four-class task.
D3. Br35H pred (60) is unlabelled and excluded.
D4. Br35H no is the only Br35H content used, contributing no-tumour images.
D5. Br35H and SARTAJ share 179 exact duplicate groups, so they are public
    collections rather than independent institutions. In Step 1.4 each shared
    image is kept once, with priority Figshare, then SARTAJ, then Br35H.
D6. SARTAJ has 23 duplicate groups spanning its Training and Testing folders,
    so its original split is discarded and rebuilt after deduplication.
D7. Exact duplicates showed no label conflicts. Any conflict found later by
    perceptual hashing removes the whole group.
D8. SHA256 cannot detect Figshare reuse because Figshare is stored as .mat.
    Step 1.4 hashes decoded, normalised pixels for all three sources.



## 2026-10-01 Workflow

D0. All work and results are committed directly to main, and dev is retired. One researcher, pull requests caused errors, and tags already pin the code for every run.

## 2026-10-01 Step 1.4 deduplication and splits

D9. Preprocessing is percentile clipping at 0.5 and 99.5, 8-bit scaling, zero padding to square, and box resizing to 128 x 128. Figshare arrays are transposed after reading because MATLAB stores them column-major.
D10. Near duplicates are pairs within Hamming distance 4 on a 64-bit DCT perceptual hash, and split units link images within distance 6 plus all slices of one Figshare patient.
D11. Figshare slices are never removed for resembling other Figshare slices. A conflicting label on a copy of a Figshare slice is resolved in favour of Figshare, and other label conflicts remove the whole group. This refines D7.
D12. Splits are 70, 10, and 20 percent for train, validation, and test, stratified by dominant source and label of each split unit, with seed 2026.
D13. Label ids are glioma 0, meningioma 1, notumor 2, pituitary 3.
D14. Dirichlet partitions require at least 20 images per client, and scenario seeds are 2026 plus the scenario index.

## 2026-10-01 Step 1.4 deduplication and splits (revision 2)

D15. Supersedes D10 and D11. The first run used union-find grouping, which chained distinct images, removed 45 Figshare slices, and merged 49.9% of the clean set into one split unit. Deduplication is now greedy. Images are visited in the order Figshare, SARTAJ, Br35H, then path, and an image is removed only if it lies within Hamming distance 4 of an image already kept. A copy whose label differs from a kept Figshare slice is removed and Figshare is kept. Any other label disagreement removes both images.
D16. Split units are single Figshare patients, plus groups of SARTAJ and Br35H images linked within distance 6 among themselves. Hash links never join Figshare patients, because patient IDs are authoritative. A unit larger than 5% of the clean set stops the pipeline.
D17. Hamming distances on this hash are always even, because every hash has the same number of set bits.

## 2026-10-01 Step 1.4 deduplication and splits (revision 3)

D18. Figshare patients sharing an identical slice within Hamming distance 4 are merged into one split unit. A shared slice implies a duplicated or misfiled patient identity, which would otherwise leak one person across splits. 48 patient pairs were merged.

## 2026-10-06 Step 2.1 pipeline

D19. Threat model. The protected unit is a single training image at each client (record-level DP). The adversary is an honest-but-curious server, or anyone who sees any released global model. Client dataset sizes and the server-held validation split are treated as public. Client-level DP is not used, because with 3 to 15 clients the noise it requires would destroy utility. delta is 1e-5, below 1/n for the 2,610-image training pool.
D20. ResNet-18 uses GroupNorm instead of BatchNorm, because BatchNorm mixes examples within a batch and breaks per-example sensitivity. Pretrained ImageNet weights are loaded, the RGB stem is summed to one channel, and BatchNorm affine terms are copied into GroupNorm. Running statistics are discarded, so this is an initialisation, not an exact transfer.
D21. Every client uses Poisson sampling with a fixed rate q = 0.1 and runs round(local_epochs / q) local steps per round. The noisy gradient sum is divided by the expected batch size q * n_j, never by the realised batch size. With a fixed q, every client spends the same epsilon. Hypothesis H5 is restated accordingly. A smaller client has a smaller expected batch, so the noise standard deviation in its averaged gradient, sigma * C / (q * n_j), grows as n_j shrinks.
D22. The reported model is the round with the lowest validation loss on the central validation split. Epsilon is reported for all T rounds, because every round's global model is released to the server. Learning-rate tuning on validation data is disclosed as a limitation and its privacy cost is not accounted.
D23. FedProx proximal terms and SCAFFOLD corrections are added after privatisation, which is post-processing and costs no privacy. Local momentum defaults to 0, so SCAFFOLD option II (Karimireddy et al., 2020) holds exactly.
D24. Per-example gradients are computed with torch.func (vmap over grad) in chunks of max_physical_batch. Opacus is used only for RDP accounting and noise calibration, cross-checked against Google dp-accounting in tests (RDP agreement within 5 percent, PLD never above RDP).

## 2026-10-06 Step 2.3 pilot findings

D25. A third model, resnet18_frozenbn, keeps torchvision BatchNorm with pretrained ImageNet statistics but holds every BatchNorm layer in eval mode permanently. In eval mode BatchNorm is a fixed per-example affine map, so no example influences another and per-example sensitivity holds. Tests check that outputs do not depend on batch composition and that running statistics never change. This replaces the BatchNorm-to-GroupNorm swap (D20) as the pretrained baseline, because that swap discarded the pretrained statistics.
D26. data.version (curated-v1) is part of every resolved config, so every run_id now encodes the dataset version. This changed all run_ids once. The 28 Step 2.3 pilot runs keep their old ids and stay archived in results/runs and results/tables/week02_*.csv as pilot evidence only.
D27. The Step 2.3 pilot underfit. Without DP, training accuracy stayed between 0.52 and 0.69, meningioma sensitivity was near 0 for the small CNN, and on natural3 no-tumour sensitivity was 0, even for ResNet. The pipeline itself is verified (21 tests, epsilon on target), so the cause is optimisation (600 plain SGD steps without momentum) plus FedAvg drift under extreme label skew. Step 2.4 adds centralised baselines, local momentum 0.9, 100 rounds, and a wider CNN before any Week 3 grid is fixed.

## 2026-10-06 Step 2.4 diagnostics

D28. Every non-DP run of resnet18_frozenbn diverged. Four produced NaN from round 1, and one collapsed to a single class. The DP runs of the same model trained well, because per-example clipping bounds every update. Frozen BatchNorm no longer renormalises activations, so unclipped gradients can explode. Non-DP runs of this model therefore use global-norm gradient clipping at 1.0, a standard stabiliser. A clip-only regime (per-example clipping, zero noise) separates clipping bias from noise cost. Runs now stop cleanly and are flagged when weights or validation loss become non-finite, instead of crashing. noise_multiplier 0 is now honoured (it was previously treated as unset). A cosine learning-rate decay across rounds is available and is data-independent.
D29. Gate A decision. resnet18_frozenbn becomes the primary model. Centralised with DP at epsilon 8 and lr 0.01, it reached balanced accuracy 0.851 (test accuracy 0.862) and was still improving at round 100. The best small CNN under DP reached 0.57, and without DP 0.73, with meningioma sensitivity at or below 0.22. The small CNN is kept only for a limited capacity ablation.

## 2026-10-06 Step 2.4 closed, settings frozen

D30. Settings frozen after Step 2.4 and written into configs/base.yaml. Model resnet18_frozenbn with ImageNet weights, SGD momentum 0.9, cosine decay over 100 rounds, one local epoch per round, Poisson rate q 0.1, clip norm 1.0, delta 1e-5, global-norm clip 1.0 for non-DP runs. Centralised and local-only runs use lr 0.02, the best of {0.003, 0.01, 0.02} with cosine decay in every privacy regime (constant lr 0.05 was worse in Step 2.4). Federated runs choose lr from {0.003, 0.01} per configuration (partition and epsilon) by the lowest validation loss at seed 0, using scripts/select_lr.py. Seed 0 serves only for selection, and seeds 1 to 5 are reported. No further tuning is planned.
D31. Step 2.4 findings that motivate the protocol. Centrally at lr 0.02, balanced accuracy is 0.918 without DP, 0.893 with clipping only, and 0.849 at epsilon 8, so clipping and noise cost about 2.5 and 4.4 points. Meningioma sensitivity falls most, from 0.90 to 0.62. Federated at epsilon 8, natural3 (large clients) reaches 0.806 while dir0.5_k10 (small clients) reaches 0.714, although natural3 is more label-skewed. This matches H5, because the per-client noise scale sigma C / (q n_j) is larger for small clients. On dir0.5_k10, lr 0.01 under DP made validation loss rise while balanced accuracy still climbed, an over-confidence effect, so lr must be chosen per partition.

## 2026-10-06 Step 1.2 literature check

D32. The literature check (docs/related_work.csv, 43 studies, docs/gap_statement.md) shows that dataset contamination in these public sources (Vangala et al. 2026), DP damage to explanations in general (Saifullah et al. 2024, Sett and Manna 2024), and the dataset-size dependence of DP (Ziller et al. 2024) are already published. Adopted without changing training settings: RQ3 is narrowed to localisation against expert masks with clipping separated from noise under federated heterogeneity, RQ4 adds patient-level membership inference and the group-privacy bound, a source-prediction probe and per-source metrics control the source-label confound, and perceptual-hash duplicates are re-checked with normalised cross-correlation. Patient-level (unit-level) DP training is proposed as an extra privacy regime and awaits a scope decision.
