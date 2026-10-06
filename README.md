# dpfl-braintumor

Differentially private federated learning for multi-class brain tumour MRI,
prepared for submission to Intelligent Oncology.

## Repository layout
| Folder | Contents |
| --- | --- |
| configs/ | YAML configs and weekly run grids |
| data_manifest/ | Source records, hashes, duplicates, frozen splits |
| src/ | Data, models, federated algorithms, privacy, XAI, attacks, evaluation |
| scripts/ | Grid runner, result pushing, aggregation, figure generation |
| tests/ | Unit tests, including DP correctness tests |
| notebooks/kaggle/ | Thin Kaggle launcher notebooks only |
| results/ | Per-run metrics and aggregated tables |
| figures/ | Script-generated figures |
| paper/ | Manuscript LaTeX source |
| docs/ | Roadmap, decisions, lab notebook, Kaggle setup |

## Data
Images are not stored here. Curated arrays live in a private Kaggle Dataset,
and data_manifest/ records every source, hash, and split needed to rebuild them.

## Reproduction

### Data (Phase 1)
Run notebooks/kaggle/01-data-acquisition, 02-dedup-splits and 03-verify-curated in order.
They rebuild the private Kaggle Dataset brain-mri-curated-v1 and its checks in data_manifest/.

### Tests
```
pip install -r requirements.txt
python -m pytest -q tests
```
Tests run on CPU with a synthetic archive that has the real schema.

### One run
```
python -m src.run --config configs/experiments/smoke.yaml --data-dir <folder with brain_mri_curated_v1.npz>
```
Each run writes results/runs/<run_id>/ with config.yaml, env.json, log.txt, predictions.csv and metrics.json.
The run_id is a hash of the fully resolved config (configs/base.yaml plus overrides).

### A grid on Kaggle
notebooks/kaggle/04-step2-pilot.ipynb clones the code at a tag, runs the test suite, and runs grids with
one process per T4 GPU. Each finished run is pushed to main. Summaries come from scripts/aggregate.py.
