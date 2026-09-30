# Project Roadmap: DP Federated Learning for Brain Tumor MRI (Intelligent Oncology)

Sep 30, 2026 · @S Ranjan

## How to Use This Roadmap

This roadmap governs 12 weeks of work toward a Research Article submission to *Intelligent Oncology*, and no step starts until the previous step passes its review.

Each step below carries an objective, a task checklist, deliverables, acceptance criteria, and a review record. The acceptance criteria are objective tests, so a reviewer can mark a step passed or failed without interpretation.

### Step Review Protocol

1. **Complete the tasks** and tick each box only when the corresponding artefact exists in GitHub.
2. **Verify every acceptance criterion** against the repository, not against memory or notebook outputs.
3. **Fill the review record** with the date, reviewer name, verdict, and the commit hash or tag reviewed.
4. **Record any deviation** in the Change Log at the end of this document and in `CHANGELOG.md` in the repository.
5. **Update the Status Dashboard** so the project state is visible at a glance.

### Verdict Definitions

| Verdict | Meaning | Next action |
| --- | --- | --- |
| Passed | Every acceptance criterion met | Tag the commit and start the next step |
| Passed with notes | Minor gaps that do not affect later steps | Log notes, fix within the next step |
| Failed | Any acceptance criterion unmet | Fix, then request a fresh review |
| Blocked | External dependency unavailable | Log in the Risk Register and escalate |

### Standing Rules

- All code, configurations, manifests, and metrics live in GitHub, and Kaggle notebooks remain thin launchers only.
- Every Kaggle run checks out a tag, never a moving branch, so each result maps to one immutable commit.
- No number enters the manuscript unless a script regenerates it from `results/` in the repository.
- Splits are frozen after Step 1.4, and any change requires a new versioned split file plus full reruns.

## Project Summary

The project rebuilds a desk-rejected ACM TOPS manuscript into a rigorous study of what formally correct differential privacy costs in accuracy, calibration, and explanation fidelity for federated brain tumour MRI classification.

| Item | Detail |
| --- | --- |
| Target journal | *Intelligent Oncology* (KeAi and Elsevier, open access) |
| Article type | Research Article with structured abstract and IMRaD structure |
| Scope match | Cancer imaging, patient privacy and data security, medical interpretability |
| Working title | Differentially Private Federated Learning for Multi-Class Brain Tumor MRI Classification and Its Effect on Diagnostic Reliability and Explanation Fidelity Across Heterogeneous Sources |
| Compute | Kaggle T4 x2, about 30 GPU hours per week |
| Code and results | Private GitHub repository `dpfl-braintumor`, archived on Zenodo at submission |
| Fee | USD 1800 APC, waived for submissions before 31 December 2027 |

### Flaws in the Previous Manuscript That This Plan Fixes

1. The noise multiplier of 0.0005 cannot yield the reported ε of 7.8, and batch-level clipping gives no per-example guarantee.
2. Every client received about 1,528 images regardless of client count, so partitions duplicated data and invalidated the scalability claim.
3. A Dirichlet α of 5.0 is close to IID, single-seed results lacked variance, and no baselines were reported.
4. Validation loss rose steadily while training accuracy reached 100%, indicating overfitting and poor calibration.
5. XAI claims lacked quantitative evaluation, and several citations were attributed to the wrong papers.

### Research Questions and Hypotheses

| ID | Question or hypothesis | Primary evidence |
| --- | --- | --- |
| RQ1 | How does record-level DP at ε of 1, 3, 8, and infinity affect accuracy, glioma sensitivity, and calibration? | Privacy and utility curves, ECE |
| RQ2 | How do label skew and source-level feature shift change FedAvg, FedProx, and SCAFFOLD under DP? | Algorithm comparison across partitions |
| RQ3 | Does DP noise degrade Grad-CAM localisation against expert tumour masks, and does heterogeneity amplify it? | Pointing game, heatmap IoU, deletion AUC |
| RQ4 | Does formal ε translate into resistance to membership inference? | Attack TPR at 0.1% FPR |
| H5 | More clients on a fixed pool shrink local data, raise the sampling rate, and raise privacy cost at fixed noise | Client count sweep with disjoint partitions |

The mechanism behind H5 follows from the per-client sampling rate and step count.

```latex
q_j = \frac{B}{n_j}, \qquad S_j = T \cdot E \cdot \frac{n_j}{B}, \qquad n_j \approx \frac{n_{\text{train}}}{N}
```

Here B is the batch size, T the number of rounds, E the local epochs, and N the number of clients.

## Status Dashboard

All 18 steps are not started, and Week 1 is assumed to begin on Monday 5 October 2026, so every target date shifts if the start moves.

| Step | Title | Week | Target end | GitHub tag | Status | Reviewed on |
| --- | --- | --- | --- | --- | --- | --- |
| 1.1 | Repository and Kaggle setup | 1 | 11 Oct | Passed | Passed | 2026-09-30 |
| 1.2 | Literature gap verification | 1 | 11 Oct | none | Not started |  |
| 1.3 | Data acquisition and provenance | 1 | 11 Oct | none | Not started |  |
| 1.4 | Deduplication, splits, Kaggle Dataset | 1 | 11 Oct | v0.1-data | Not started |  |
| 2.1 | Core pipeline implementation | 2 | 18 Oct | none | Not started |  |
| 2.2 | DP correctness verification | 2 | 18 Oct | v0.2-pipeline-verified | Not started |  |
| 2.3 | Timing pilot and budget recalibration | 2 | 18 Oct | none | Not started |  |
| 3.1 | FedAvg core grid and baselines | 3 | 25 Oct | v0.3-week03 | Not started |  |
| 3.2 | FedAvg completion and Gate A | 4 | 1 Nov | v0.4-week04 | Not started |  |
| 3.3 | FedProx and SCAFFOLD grid | 5 | 8 Nov | v0.5-week05 | Not started |  |
| 3.4 | Client count sweep and ResNet-18 | 6 | 15 Nov | v0.6-week06 | Not started |  |
| 4.1 | Membership inference audit | 7 | 22 Nov | none | Not started |  |
| 4.2 | XAI fidelity and Gate B | 7 | 22 Nov | v0.7-week07 | Not started |  |
| 4.3 | Results freeze | 8 | 29 Nov | v0.9-results-frozen | Not started |  |
| 5.1 | Methods and Results drafting | 9 | 6 Dec | none | Not started |  |
| 5.2 | Introduction, Discussion, front matter | 10 | 13 Dec | none | Not started |  |
| 5.3 | Internal expert review | 11 | 20 Dec | v0.95-reviewed | Not started |  |
| 6.1 | Compliance check and submission | 12 | 27 Dec | v1.0-submission | Not started |  |

## Phase 1 Foundations (Week 1, 0 GPU hours)

Phase 1 produces a frozen, deduplicated, patient-level dataset and a working repository without spending any GPU quota.

### Step 1.1 Repository and Kaggle Setup

**Objective.** Establish GitHub as the single source of truth and connect Kaggle to it securely.

- [ ] Create private repository `dpfl-braintumor` with branches `main` and `dev`
- [ ] Add the folder layout `configs/`, `data_manifest/`, `src/`, `scripts/`, `tests/`, `notebooks/kaggle/`, `results/`, `figures/`, `paper/`, `docs/`
- [ ] Add `README.md`, `CHANGELOG.md`, `docs/decisions.md`, `docs/lab_notebook.md`, `docs/kaggle_setup.md`
- [ ] Add `.gitignore` excluding images, checkpoints, `.env`, and tokens
- [ ] Create a fine-grained GitHub token scoped to this repository only, with an expiry date
- [ ] Store the token as Kaggle User Secret `kaggle_dpfl-braintumor` and complete Kaggle phone verification
- [ ] Write `scripts/push_results.sh` with pull, rebase with autostash, push, and a three-attempt retry loop, run from a results copy on dev while code runs from a tag checkout
- [ ] Commit this roadmap as `docs/ROADMAP.md`

**Deliverables.** Repository skeleton, launcher notebook template, push script.

**Acceptance criteria.**

1. A private Kaggle notebook clones the repository, writes a test file, and pushes it back without printing the token.
2. Kaggle weekly GPU quota, session limit, and T4 x2 billing rules are recorded in `docs/kaggle_setup.md` with the check date.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 1.2 Literature Gap Verification

**Objective.** Confirm that RQ3 and H5 are not already answered before any GPU quota is spent.

- [ ] Search Scopus, PubMed, IEEE Xplore, and Google Scholar for 2023 to 2026 work on DP with federated learning for brain tumour MRI
- [ ] Search specifically for studies measuring how DP noise affects saliency or Grad-CAM fidelity
- [ ] Read Gupta et al. (2026, IEEE Access) and Kaissis et al. (2021, Nature Machine Intelligence) in full
- [ ] Build `docs/related_work.csv` with columns for task, dataset, FL method, DP type, ε reported, XAI method, XAI metric
- [ ] Correct every citation error found in the previous manuscript

**Deliverables.** Related work matrix of at least 25 studies, a one-page gap statement in `docs/decisions.md`.

**Acceptance criteria.**

1. The gap statement names at least one question that no study in the matrix answers.
2. If RQ3 or H5 is already answered, the research questions are revised and logged before Step 2.1.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 1.3 Data Acquisition and Provenance

**Objective.** Rebuild the dataset from its three original public sources with full provenance.

- [ ] Download Figshare (Cheng et al.) with patient IDs and tumour masks
- [ ] Download the SARTAJ and Br35H datasets
- [ ] Record URL, licence, version, download date, and file count per source in `data_manifest/sources.csv`
- [ ] Confirm whether each licence permits redistribution, and note the answer
- [ ] Store raw files in a private Kaggle Dataset `brain-mri-raw-v1`

**Deliverables.** `sources.csv`, private raw Kaggle Dataset.

**Acceptance criteria.**

1. File counts per source and class match the counts in `sources.csv` exactly.
2. Every Figshare image maps to a patient ID and a mask file.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 1.4 Deduplication, Splits, and Frozen Kaggle Dataset

**Objective.** Remove duplicates, create patient-level disjoint splits, and freeze every partition used in the study.

- [ ] Compute SHA256 and perceptual hashes for every image and save `data_manifest/hashes.csv`
- [ ] Remove exact and near duplicates below a fixed Hamming threshold, and log them in `duplicates.csv`
- [ ] Split Figshare by patient, and split SARTAJ and Br35H after deduplication
- [ ] Generate client assignments for the natural three-source scenario, α of 0.1, 0.5, 5.0 with 10 clients, IID with 10 clients, and 3, 5, 10, 15 clients at α of 0.5
- [ ] Save everything in `data_manifest/splits_v1.csv`
- [ ] Export preprocessed 128 × 128 grayscale arrays as a private Kaggle Dataset `brain-mri-curated-v1`
- [ ] Draw the data flow diagram counting images at every stage

**Deliverables.** Hash, duplicate, and split manifests, curated Kaggle Dataset, flow diagram.

**Acceptance criteria.**

1. No perceptual hash appears in both training and test sets.
2. No Figshare patient appears in more than one split.
3. For every scenario, the sum of client dataset sizes equals the training pool size exactly.
4. Array hashes in the Kaggle Dataset match `hashes.csv`.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Phase 2 Pipeline and Verification (Week 2, 8 GPU hours)

Phase 2 delivers a verified DP federated pipeline, and no experiment grid runs until every DP test passes.

### Step 2.1 Core Pipeline Implementation

**Objective.** Implement a lean, fully configurable federated training stack in PyTorch.

- [ ] Port from TensorFlow to PyTorch with pinned versions in `requirements.txt`
- [ ] Implement a small CNN with a global average pooling head, targeting about 100 thousand parameters
- [ ] Implement ResNet-18 with GroupNorm, since BatchNorm breaks per-example sensitivity
- [ ] Implement FedAvg with sample-weighted aggregation, FedProx, and SCAFFOLD in `src/fl/`
- [ ] Implement per-example clipping with Opacus or `torch.func.vmap`, plus Poisson sampling
- [ ] Keep one persistent RDP or PRV accountant per client across all rounds
- [ ] Preload curated arrays into GPU memory as float16 tensors
- [ ] Write `src/run.py` so each run is defined by one YAML config and identified by its config hash
- [ ] Log commit hash, Kaggle image, GPU name, seed, σ, q, steps, δ, and ε spent to `env.json` and `metrics.json`
- [ ] Write `scripts/run_grid.py` with GPU sharding and automatic skipping of finished runs
- [ ] Add early stopping on validation loss and checkpoint saving every ten rounds for long runs

**Deliverables.** Tested training stack, grid runner, launcher notebook.

**Acceptance criteria.**

1. A 5-round smoke run completes on Kaggle and pushes a complete run folder to GitHub.
2. Interrupting and relaunching a grid skips finished runs and completes only the missing ones.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 2.2 DP Correctness Verification

**Objective.** Prove the privacy mechanism is implemented correctly before generating any reported result.

- [ ] Clipping test confirms every per-example gradient norm is at most C after clipping
- [ ] Accountant test reproduces ε for a known configuration with a second, independent accountant library
- [ ] Zero-noise test confirms σ of 0 reproduces non-private training exactly under a fixed seed
- [ ] Memorisation test confirms training accuracy stays well below 100% at ε of 8 or lower
- [ ] Noise calibration script solves σ for each target ε given q, steps, and δ below 1 divided by the client data size
- [ ] Document the threat model and record-level privacy unit in `docs/decisions.md`

**Deliverables.** Passing test suite in `tests/`, σ calibration table for every planned configuration.

**Acceptance criteria.**

1. All four DP tests pass in continuous integration or in a logged Kaggle run.
2. The two accountants agree on ε within 5% for every calibration configuration.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 2.3 Timing Pilot and Budget Recalibration

**Objective.** Measure real run times so the GPU budget rests on data rather than estimates.

- [ ] Run one configuration per model and algorithm with and without DP, three repeats each
- [ ] Confirm that two parallel shards on T4 x2 do not exhaust GPU memory
- [ ] Record median minutes per run in `docs/lab_notebook.md`
- [ ] Recompute the GPU Budget Ledger and adjust seed counts if the total exceeds 90 quota hours

**Deliverables.** Timing table, updated ledger.

**Acceptance criteria.**

1. The recalibrated total fits within weeks 3 to 8 at no more than 20 quota hours per week.
2. Any seed reduction affects secondary baselines only, never the FedAvg core grid.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Phase 3 Experiment Grids (Weeks 3 to 6, about 60 GPU hours)

Phase 3 runs about 230 atomic experiments on two parallel T4 shards, pushing results to GitHub after every run.

Every grid step follows the same launch routine.

1. Tag the verified commit and check it out in the launcher notebook.
2. Start two shards with `CUDA_VISIBLE_DEVICES` set to 0 and 1.
3. Use Save and Run All for background execution of long grids.
4. After the session, confirm the expected run count in `results/runs/` and log quota hours used.

### Step 3.1 FedAvg Core Grid and Baselines (Week 3, 16 GPU hours)

**Objective.** Answer RQ1 for FedAvg and establish the non-federated reference points.

- [ ] FedAvg at ε of infinity and 8 across natural, α 0.1, α 0.5, α 5.0, and IID partitions, five seeds each
- [ ] Centralised training with and without DP, five seeds each
- [ ] Local-only training at each natural source, five seeds each
- [ ] Run `aggregate.py` in a CPU session and inspect preliminary tables

**Acceptance criteria.**

1. Every planned run folder exists with a complete `metrics.json` and the correct tag hash.
2. Reported ε in each DP run matches its calibration target within 5%.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 3.2 FedAvg Completion and Gate A (Week 4, 12 GPU hours)

**Objective.** Complete the FedAvg privacy sweep and decide the primary model.

- [ ] FedAvg at ε of 3 and 1 across all partitions, five seeds each
- [ ] Plot accuracy and glioma sensitivity against ε for each partition
- [ ] Apply Gate A and log the decision in `docs/decisions.md`

**Acceptance criteria.**

1. The FedAvg results table has mean, standard deviation, and bootstrap 95% confidence intervals for every cell.
2. Gate A has a recorded decision with its supporting numbers.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 3.3 FedProx and SCAFFOLD Grid (Week 5, 14 GPU hours)

**Objective.** Answer RQ2 by comparing drift-correcting algorithms under DP.

- [ ] FedProx with μ of 0.01 and 0.1, choosing μ on validation data only
- [ ] SCAFFOLD with control variates saved in checkpoints
- [ ] Both algorithms at ε of infinity and 3 across all partitions, five seeds each

**Acceptance criteria.**

1. Paired comparisons against FedAvg report McNemar tests and effect sizes.
2. The μ selection procedure is documented and does not touch the test set.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 3.4 Client Count Sweep and ResNet-18 (Week 6, 18 GPU hours)

**Objective.** Test H5 and measure the benefit of public pretraining under DP.

- [ ] FedAvg with 3, 5, 10, and 15 clients at α of 0.5, ε of infinity and 3, five seeds each
- [ ] Record q, steps, and σ per client count to show the privacy cost mechanism directly
- [ ] ResNet-18 pretrained on ImageNet at all four privacy levels on the natural partition, three seeds each
- [ ] Optionally log energy with CodeCarbon for the sustainability note

**Acceptance criteria.**

1. Client datasets are disjoint and sum to the training pool for every client count.
2. A table links N, local data size, q, σ, and accuracy for H5.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Phase 4 Audit, Explainability, and Freeze (Weeks 7 to 8, about 15 GPU hours)

Phase 4 turns trained models into evidence for RQ3 and RQ4, then freezes every result that the manuscript will report.

### Step 4.1 Membership Inference Audit (Week 7, 8 GPU hours)

**Objective.** Answer RQ4 by testing whether formal ε translates into measured protection.

- [ ] Upload the audited checkpoints as GitHub Release assets, one seed per configuration
- [ ] Run a loss-threshold attack on every privacy level
- [ ] Train a reduced set of shadow models for a LiRA-style attack on the natural partition
- [ ] Report true positive rate at 0.1% false positive rate, with ROC curves on a log scale

**Acceptance criteria.**

1. Attack results exist for ε of infinity, 8, 3, and 1 with confidence intervals.
2. Member and non-member sets are disjoint and documented in the repository.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 4.2 XAI Fidelity and Gate B (Week 7, 4 GPU hours plus CPU)

**Objective.** Answer RQ3 with quantitative explanation metrics against expert tumour masks.

- [ ] Generate Grad-CAM, Grad-CAM++, and Integrated Gradients maps on the Figshare test patients
- [ ] Compute pointing-game accuracy and top 15% heatmap IoU against masks in a CPU session
- [ ] Compute deletion and insertion AUC for faithfulness
- [ ] Run model-parameter randomisation sanity checks
- [ ] Plot each metric against ε and α
- [ ] Apply Gate B and log the decision

**Acceptance criteria.**

1. Each metric is reported per privacy level with bootstrap 95% confidence intervals.
2. Sanity checks confirm that maps change when model weights are randomised.
3. The skull-border artefact from the previous manuscript is either absent or explicitly analysed.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 4.3 Results Freeze (Week 8, 3 GPU hours)

**Objective.** Lock every number, figure, and table that the manuscript will report.

- [ ] Rerun any failed or missing configurations
- [ ] Compute ECE, Brier score, reliability diagrams, DeLong tests, and McNemar tests
- [ ] Generate every figure and table with `make_figures.py` from the results folder
- [ ] Write `results/SUMMARY.md` listing each headline finding with its file source
- [ ] Tag `v0.9-results-frozen` and create a GitHub Release

**Acceptance criteria.**

1. Deleting `figures/` and running one command regenerates every figure identically.
2. No planned run is missing from the aggregated tables.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Phase 5 and 6 Writing and Submission (Weeks 9 to 12, 0 to 5 GPU hours)

Phase 5 and 6 write a fully new manuscript in the Elsevier CAS LaTeX template inside `paper/`, reusing no prose from the ACM version.

### Step 5.1 Methods and Results Drafting (Week 9)

**Objective.** Write the sections that depend only on frozen results.

- [ ] Section 2 Materials and methods covering sources, curation, ethics, federation design, DP accounting, models, XAI evaluation, privacy audit, and statistics
- [ ] Section 3 Results in the order RQ1, RQ2, H5, RQ3, RQ4, calibration
- [ ] Every number cross-checked against `results/SUMMARY.md`
- [ ] Equations as editable text and figures at 300 dpi or higher

**Acceptance criteria.**

1. Methods contain enough detail for an independent group to reproduce every run from the repository.
2. Every Results sentence with a number cites a figure, table, or supplementary item.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 5.2 Introduction, Discussion, and Front Matter (Week 10)

**Objective.** Frame the findings for an oncology readership and complete all front matter.

- [ ] Introduction of about 900 words without a detailed literature survey, as the guide requires
- [ ] Discussion covering clinical interpretation, comparison with prior work, limitations, and deployment implications
- [ ] Structured abstract with Background, Methods, Results, and Conclusions, including effect sizes and confidence intervals
- [ ] Contributions paragraph mapped one to one onto the abstract, with integrated findings
- [ ] Up to six keywords, three to five highlights of at most 85 characters, and a graphical abstract
- [ ] Related work comparison table moved to supplementary material

**Acceptance criteria.**

1. Each research question in the Introduction receives a direct answer in the Conclusions.
2. The abstract stands alone without references or undefined abbreviations.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 5.3 Internal Expert Review (Week 11)

**Objective.** Catch clinical and privacy errors before the journal does.

- [ ] Review by a radiologist or neuro-oncologist for clinical framing and figure interpretation
- [ ] Review by a DP specialist for the threat model and accounting
- [ ] Run a similarity check if institutional access to iThenticate or Turnitin exists
- [ ] Resolve every comment and log responses in `docs/internal_review.md`

**Acceptance criteria.**

1. No unresolved major comment remains.
2. Any rerun triggered by review fits within the 5-hour GPU reserve for this week.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

### Step 6.1 Compliance Check and Submission (Week 12)

**Objective.** Submit a manuscript that passes every editorial screening check of *Intelligent Oncology*.

- [ ] Ethics statement declaring de-identified public data with approval waived
- [ ] CRediT author statement for every author
- [ ] Competing interests form on the journal .docx template, not converted
- [ ] Funding statement or the standard no-funding sentence
- [ ] Data and code availability statement pointing to the Zenodo DOI and original data sources
- [ ] Generative AI declaration before the references, if any tool assisted language editing
- [ ] AMA numbered superscript references with DOIs, arXiv replaced by published versions wherever possible
- [ ] Two suggested reviewers with institutional emails, outside JNU, with no recent collaboration
- [ ] Corresponding author email, full postal address, and telephone with country code
- [ ] Enable GitHub and Zenodo integration, tag `v1.0-submission`, and record the DOI
- [ ] Submit through Editorial Manager and record the manuscript ID here

**Acceptance criteria.**

1. Every item in the journal submission checklist is ticked.
2. The Zenodo archive matches the tagged commit exactly.

| Date | Reviewer | Verdict | Commit or tag | Notes |
| --- | --- | --- | --- | --- |
|  |  |  |  |  |

## Decision Gates, GPU Ledger, and Storage Policy

Two decision gates can change the plan, the GPU plan totals 83 to 88 quota hours against a weekly ceiling of 30, and every artefact has one fixed storage location.

### Decision Gates

| Gate | When | Test | If met | If not met |
| --- | --- | --- | --- | --- |
| Gate A | End of Step 3.2 | Small CNN FedAvg accuracy at ε of 3 stays within 10 points of the non-private model on the natural partition | Keep the small CNN as primary model | Promote pretrained ResNet-18 to primary and rerun its grid in Step 3.4 |
| Gate B | End of Step 4.2 | At least one fidelity metric changes monotonically with ε beyond its confidence interval | Keep RQ3 as a headline finding | Report RQ3 as a null result and lead with RQ1, RQ2, and H5 |

### GPU Budget Ledger

The actual hours column is filled after each Kaggle session, and any week exceeding its plan by more than 25% triggers a review of the remaining grid.

| Week | Step | Planned quota hours | Actual quota hours | Runs planned | Runs completed |
| --- | --- | --- | --- | --- | --- |
| 1 | 1.1 to 1.4 | 0 |  | 0 |  |
| 2 | 2.1 to 2.3 | 8 |  | about 20 |  |
| 3 | 3.1 | 16 |  | 70 |  |
| 4 | 3.2 | 12 |  | 50 |  |
| 5 | 3.3 | 14 |  | 80 |  |
| 6 | 3.4 | 18 |  | 52 |  |
| 7 | 4.1 and 4.2 | 12 |  | about 16 |  |
| 8 | 4.3 | 3 |  | reruns |  |
| 9 to 12 | 5.1 to 6.1 | 0 to 5 |  | reruns |  |
| **Total** |  | **83 to 88** |  |  |  |

### Artefact Storage Policy

| Artefact | Location | Reason |
| --- | --- | --- |
| Code, configs, tests, roadmap | GitHub repository | Versioned and permanent |
| Raw and curated images | Private Kaggle Datasets plus one institutional backup | Size and licensing |
| Hashes, duplicates, splits | GitHub `data_manifest/` | Exact reconstruction of every split |
| Per-run metrics, logs, environment | GitHub `results/runs/` | Small and fully traceable |
| Routine checkpoints | Kaggle notebook outputs | Temporary and regenerable |
| Checkpoints used for XAI and attacks | GitHub Release assets | Needed for revision and review |
| Figures and tables | GitHub, regenerated by script | Never edited by hand |
| Manuscript source | GitHub `paper/` | Linked to the results it reports |
| Submission archive | Zenodo DOI from `v1.0-submission` | Citable and permanent |

Git LFS is not used, because its free storage and bandwidth allowances are small.

## Risk Register, Change Log, and Open Questions

Six risks can delay submission, and each has an owner-free trigger and a predefined response.

### Risk Register

| Risk | Trigger | Likelihood | Response |
| --- | --- | --- | --- |
| Kaggle quota or session limits change | Settings page differs from `docs/kaggle_setup.md` | Medium | Recompute ledger, reduce secondary baseline seeds from five to three |
| DP runtime far exceeds estimates | Step 2.3 median run time above 30 minutes | Medium | Shrink rounds, drop IID reference, keep FedAvg core grid intact |
| Utility collapses under correct DP | Gate A fails | Medium | Switch to pretrained ResNet-18, report the cost honestly |
| Novelty already published | Step 1.2 finds RQ3 or H5 answered | Low to medium | Revise questions before Step 2.1 |
| Source licences forbid redistribution | Step 1.3 licence check | Medium | Keep arrays private, point data statement to original sources |
| Token leak or account loss | Token printed or Kaggle access lost | Low | Revoke token immediately, restore from GitHub and institutional backup |

### Change Log

Every deviation from this roadmap is logged here and mirrored in `CHANGELOG.md`, newest entry first.

| Date | Step | Change | Reason | Approved by |
| --- | --- | --- | --- | --- |
| 30 Sep 2026 | 1.1 | Push script uses rebase with autostash | Rebase fails when a crashed run leaves unstaged files |  |
| 30 Sep 2026 | 1.1 | Launchers use two working copies, code at a tag and results on dev | A tag checkout is detached and cannot push results |  |
| 30 Sep 2026 | 1.1 | Kaggle secret label is kaggle\_dpfl-braintumor instead of GH\_TOKEN | Label chosen at setup, kept for simplicity |  |
| 30 Sep 2026 | 1.1 | .gitignore pattern changed from data/ to /data/ | The unanchored pattern silently ignored src/data |  |
| 30 Sep 2026 | All | Roadmap created | Revision plan for *Intelligent Oncology* |  |

### Open Questions

- [ ] Confirm the current Kaggle weekly GPU quota and maximum session length before Step 1.1
- [ ] Confirm the indexing status of *Intelligent Oncology* against thesis and institutional requirements
- [ ] Identify a radiologist or neuro-oncologist willing to act as clinical reviewer or co-author
- [ ] Decide whether a formal reader study is worth the ethics approval time
- [ ] Confirm SARTAJ glioma label quality, and decide whether to exclude or relabel doubtful images
- [ ] Choose a fallback venue, such as Computers in Biology and Medicine or IEEE JBHI
