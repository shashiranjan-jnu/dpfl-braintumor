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
Steps are added as each roadmap phase is completed.
