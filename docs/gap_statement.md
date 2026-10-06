# Step 1.2 Gap Statement

Date 2026-10-06. Matrix in `docs/related_work.csv` (43 studies, seven groups). Searches covered PubMed, arXiv, publisher pages and Google-indexed literature for 2019 to October 2026, on federated learning for brain tumour MRI, differential privacy in medical imaging, the interaction of privacy and explainability, dataset contamination in public brain tumour data, and patient-level privacy risk.

## 1. What the literature already answers

1. **Public brain tumour benchmarks are contaminated.** Vangala et al. (2026, arXiv 2610.00421) report that 28.8% of the Kaggle Nickparvar test split has a near-twin in training, that 95.5% of traceable Figshare test images share a patient with training, and that file metadata alone reaches 0.959 balanced accuracy for tumour versus no tumour, a source-label confound. Our Step 1.4 deduplication is therefore a necessary substrate, not a headline contribution.
2. **DP degrades gradient-based explanations in general.** Saifullah et al. (2024, Frontiers in AI) show that DP hampers attribution quality across six image datasets and that FedAvg moderates the damage. Sett and Manna (2024, arXiv 2412.20798) find on chest X-rays that DP and non-private attributions barely agree (Privacy Invariance Score at most 0.32). The original RQ3, "does DP degrade explanations", is answered at a general level.
3. **DP utility depends on dataset size.** Ziller et al. (2024, Nature Machine Intelligence) show small datasets lose far more accuracy at a given epsilon. The mechanism behind H5 (noise scale sigma C over q n_j) is standard DP-SGD arithmetic, so H5 cannot be framed as a new mechanism.
4. **DP hits difficult classes hardest.** Bagdasaryan and Shmatikov (2019) and Tayebi Arasteh et al. (2024) report that underrepresented classes and difficult diagnoses suffer most. Our meningioma result (0.90 to 0.62 sensitivity) is an instance, not a discovery.
5. **Aggregate privacy metrics hide patient-level risk.** Knolle et al. (2026, Nature) show near-perfect membership inference for individual patients with several similar records, even when the aggregate attack looks random.

## 2. What remains open

**G1. Formally accounted DP on a leakage-free brain tumour benchmark.** Every federated brain tumour classification study found reports 92 to 99.6% accuracy on merged public data without deduplication or patient-level splitting (Albalawi 2024, Mastoi 2025, Muntaqim 2025, Panwar 2026, Helal 2026, Kukreja 2024). The few that add DP (Lakshmi Vasanthi 2026, Qazi 2026) do not state epsilon, delta or the privacy unit in the abstract. No study found reports a privacy-utility curve with RDP accounting on a deduplicated, patient-disjoint split.

**G2. Explanation fidelity measured against expert tumour masks, with clipping separated from noise.** Saifullah et al. and Sett and Manna measure fidelity or agreement, not localisation against expert annotations, and neither uses brain MRI or examines label-skewed federated partitions. To our knowledge from the papers read, no study separates the effect of per-example clipping from the effect of noise on explanations. Our clip-only regime makes this decomposition possible at no extra design cost.

**G3. Client size versus label skew under DP in federated learning.** The mechanism is known, but no medical imaging study found isolates empirically that, under DP, client size can dominate label skew. Our Step 2.4 pilot already shows this (natural3 0.806 versus dir0.5_k10 0.714 at epsilon 8 despite stronger label skew in natural3). This is a supporting empirical finding, not a headline.

**G4. Patient-level privacy for slice-based brain MRI.** Figshare contributes 13.2 slices per patient on average. Record-level DP at epsilon 8 gives a patient with k slices only the group-privacy bound of roughly k times epsilon, which is vacuous for k near 13. No brain MRI federated study found audits patient-level membership inference or trains with patient-level (unit-level) DP. Knolle et al. (2026) establish why this matters but do not study federated brain MRI.

**G5. Source-label confound inside federated partitions.** Vangala et al. show the confound for centralised benchmarks. In a source-based federated partition (natural3), the no-tumour class comes only from SARTAJ and Br35H, so a model may learn the source rather than the pathology. No federated study found controls for this.

## 3. Consequences for the study (decision D32)

Adopted now, at no change to the frozen training settings:

1. **RQ3 narrowed.** Measure Grad-CAM localisation against Figshare expert masks (pointing game, top-k IoU) and faithfulness (deletion and insertion), and separate clipping from noise using the clip-only regime, across natural and Dirichlet partitions. Prior work is cited as establishing the general degradation.
2. **RQ4 upgraded to patient level.** Report membership inference per Figshare patient (aggregating over the patient's slices) next to the per-image attack, and report the group-privacy bound for the median patient.
3. **Source confound controls.** Add a source-prediction probe on penultimate features, report per-source metrics, and report tumour-versus-no-tumour performance within SARTAJ, the only source holding all four classes.
4. **Deduplication cross-check.** Re-verify the perceptual-hash duplicates with normalised cross-correlation on 64 x 64 arrays (the method of Vangala et al.), because they report false positives for perceptual hashing at their settings. CPU only.
5. **Positioning.** Deduplication and the 92 to 99% literature become motivation in the Introduction, with Vangala et al. credited.

Proposed, needs an explicit decision because it changes scope and GPU budget:

6. **Patient-level (unit-level) DP training.** Sample split units, which are Figshare patients or SARTAJ and Br35H look-alike groups, instead of single images, and clip each unit's summed gradient. This gives a guarantee per patient rather than per slice and directly addresses G4. Estimated cost is one extra privacy regime in the federated seeds grid, roughly 40 runs or 6 to 8 GPU hours.

## 4. Verification status

Every row in the matrix lists how it was verified. Eight method references and two other rows (Gupta 2026 from the original manuscript, the MTA-Swin title) still need their DOIs or full texts checked before submission. Gupta et al. (2026) must be read in full, because it is the closest prior work cited in the ACM version.
