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
