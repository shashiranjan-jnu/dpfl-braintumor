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
