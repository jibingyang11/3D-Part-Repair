# 3D Part Repair: Target-Aware Local Mesh Repair for Semantic Part Removal

This repository provides the source code, dataset construction pipeline, evaluation
benchmark, and trained models for the paper:

> **Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change**
>
> Submitted to *The Visual Computer*

This code is directly related to the manuscript submitted to *The Visual Computer*.
Please cite this work when using the provided resources.

## Overview

This project introduces a new task: **minimal-change local mesh repair after semantic
part removal**, which closes only the opening caused by part deletion while preserving
unaffected regions. The key contributions include:

1. Clear definition of the novel mesh repair task
2. Removed-part-aware target boundary loop selection (geometric + learned)
3. **Lightweight MLP-based patch generation** extending beyond geometric baselines
4. Comprehensive five-dimensional evaluation protocol
5. **Multi-category benchmark** (Chair, Table, StorageFurniture)
6. Extensive comparison with SOTA methods

## Repository Structure

```
3D-Part-Repair/
├── configs/                    # YAML configuration files
│   ├── default.yaml           # Default configuration
│   ├── chair_leg.yaml         # Chair leg removal config
│   └── multi_category.yaml   # Multi-category experiment config
├── data/
│   ├── raw/                   # Raw paired data (generated)
│   ├── processed/             # Processed data
│   └── splits/                # Train/val/test splits
├── notebooks/                 # Jupyter notebooks (run in order)
│   ├── 00_env_check.ipynb
│   ├── 01_build_dataset.ipynb
│   ├── 02_extract_boundary_loops.ipynb
│   ├── 03_target_scoring_baseline.ipynb
│   ├── 04_repair_backends.ipynb
│   ├── 05_evaluation.ipynb
│   ├── 06_ablation_target_selection.ipynb
│   ├── 07_multicategory_experiments.ipynb    # Multi-category (Table, StorageFurniture)
│   ├── 08_make_tables_and_figures.ipynb
│   ├── 09_sota_comparison.ipynb
│   ├── 10_learning_selection.ipynb
│   └── 11_mlp_patch_training.ipynb          # MLP patch training & evaluation (NEW)
├── outputs/                   # All outputs (generated)
│   ├── figures/
│   ├── tables/
│   ├── metrics/
│   ├── models/                # Trained MLP models
│   └── repaired_meshes/
├── src/                       # Source code
│   ├── baselines/             # SOTA comparison methods
│   │   ├── advancing_front.py
│   │   ├── open3d_fill.py
│   │   └── trimesh_fill.py
│   ├── data/                  # Dataset construction & loading
│   │   ├── dataset_builder.py
│   │   ├── dataset_index.py
│   │   └── sample_loader.py
│   ├── evaluation/            # Five-dimensional evaluation
│   │   ├── evaluator.py
│   │   ├── metrics_closure.py
│   │   ├── metrics_complexity.py
│   │   ├── metrics_distance.py
│   │   ├── metrics_locality.py
│   │   └── metrics_quality.py
│   ├── experiments/           # Experiment runners
│   │   ├── run_single.py
│   │   ├── run_batch.py
│   │   └── summarize.py
│   ├── geometry/              # Geometric utilities
│   │   ├── bbox.py
│   │   ├── boundary.py
│   │   ├── projection.py
│   │   ├── quality.py
│   │   └── triangulation.py
│   ├── io/                    # I/O utilities
│   │   ├── mesh_io.py
│   │   └── meta_io.py
│   ├── repair/                # Repair methods
│   │   ├── center_fan.py      # Center-fan baseline
│   │   ├── planar_patch.py    # Planar triangulation baseline
│   │   ├── mlp_patch.py       # MLP-based patch generation (NEW)
│   │   ├── minimal_area_patch.py
│   │   └── registry.py
│   ├── target_selection/      # Target loop selection
│   │   ├── features.py
│   │   ├── labeling.py
│   │   ├── learning.py        # RF/MLP/GBDT classifiers
│   │   ├── ranking.py
│   │   ├── scorers.py
│   │   └── selectors.py
│   ├── visualization/
│   │   ├── mesh_renderer.py
│   │   └── plot_utils.py
│   ├── config.py
│   └── utils.py
└── requirements.txt
```

## Installation

### Prerequisites

- Python >= 3.8
- PartNet data v0 (download from https://partnet.cs.stanford.edu/)

### Setup

```bash
# Clone the repository
git clone https://github.com/jibingyang11/3D-Part-Repair.git
cd 3D-Part-Repair

# Install dependencies
pip install -r requirements.txt

# Configure PartNet path
# Edit configs/default.yaml and set paths.partnet_root to your PartNet data_v0 directory
```

## Quick Start

Run the notebooks in order from `notebooks/` directory:

```bash
cd notebooks
jupyter notebook
```

### Notebook Execution Order

| Notebook | Purpose |
|----------|---------|
| `00_env_check` | Verify environment and dependencies |
| `01_build_dataset` | Build chair-leg removal dataset from PartNet |
| `02_extract_boundary_loops` | Extract and visualize boundary loops |
| `03_target_scoring_baseline` | Target boundary loop scoring |
| `04_repair_backends` | Run all repair baselines |
| `05_evaluation` | Evaluate repair results (5 dimensions) |
| `06_ablation_target_selection` | Ablation: RPA vs largest-hole-only |
| `07_multicategory_experiments` | **Multi-category experiments (Table, StorageFurniture)** |
| `08_make_tables_and_figures` | Generate paper tables and figures |
| `09_sota_comparison` | SOTA comparison (adv-front, Poisson, trimesh) |
| `10_learning_selection` | Learning-based target loop selection (RF/MLP/GBDT) |
| `11_mlp_patch_training` | **MLP-based patch generation training & evaluation** |

### Key New Features

#### MLP-Based Patch Generation (Notebook 11)

The MLP patch generator extends beyond geometric baselines by learning to predict
optimal patch center positions from boundary loop features:

```python
from src.repair.mlp_patch import MLPPatchGenerator, train_mlp_patch_generator

# Train
generator = train_mlp_patch_generator(
    sample_ids=train_ids,
    data_dir="data/raw",
    save_path="outputs/models/mlp_patch_generator.pkl"
)

# Use in repair
from src.repair.mlp_patch import mlp_patch_repair
result = mlp_patch_repair(damaged_mesh, target_loops, generator=generator)
```

#### Multi-Category Experiments (Notebook 07)

Extends the benchmark to Table (leg removal) and StorageFurniture (door removal):

```python
# Configure in configs/multi_category.yaml
categories:
  - category: "Chair"
    semantic_label: "leg"
    total_samples: 100
  - category: "Table"
    semantic_label: "leg"
    total_samples: 50
  - category: "StorageFurniture"
    semantic_label: "door"
    total_samples: 50
```

## Repair Methods

| Method | Type | Description |
|--------|------|-------------|
| Center-fan + RPA | Ours | Geometric centroid fan patch |
| Planar + RPA | Ours | Delaunay triangulation on fitted plane |
| **MLP Patch + RPA** | **Ours (learning)** | **MLP-predicted center offset for fan patch** |
| Advancing-front + RPA | SOTA | Classical Liepa-style advancing front |
| Poisson recon | SOTA | Global Poisson surface reconstruction |
| Trimesh fill-all | SOTA | Fill all holes indiscriminately |
| Planar + LH-only | Ablation | Planar with largest-hole target selection |

## Evaluation Metrics

1. **Closure**: Target opening sealed?
2. **Complexity**: New vertices/faces added
3. **Quality**: Triangle quality (normalized ratio)
4. **Locality**: Patch within repair zone?
5. **Distance**: Chamfer/Hausdorff to ground truth

## Citation

If you use this code or dataset, please cite:

```bibtex
@article{ji2026targetaware,
  title={Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change},
  author={Ji, Bingyang and Gao, Changxin and Chen, Fuhao and Cui, Shan},
  journal={The Visual Computer},
  year={2026}
}
```

## License

This project is released for academic research purposes.
