# Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change

> **Note:** This repository contains the source code and experiments for the manuscript
> currently submitted to *The Visual Computer*.
> If you use this code, please cite the corresponding manuscript.

## Overview

This project addresses **minimal-change local mesh repair following semantic part removal**.
Given a complete triangle mesh, we remove a semantic component (e.g., a chair leg), and the
goal is to locally close the resulting opening while minimizing geometric and topological
changes to unaffected regions.

## Key Contributions

1. **Task Definition**: Minimal-change local mesh repair after semantic part removal.
2. **Dataset**: Chair-leg removal dataset from PartNet raw mesh data.
3. **Removed-Part-Aware Target Selection**: Boundary loop selection using removed part bounding box.
4. **Learning-Based Selection**: RF / MLP / GBDT classifiers for learned target loop selection.
5. **Evaluation Protocol**: Six dimensions — closure, complexity, quality, locality, Chamfer distance, Hausdorff distance.
6. **SOTA Comparison**: Benchmarked against trimesh fill-all, advancing-front, and Poisson reconstruction.

## Installation

```bash
pip install -r requirements.txt
# Edit configs/default.yaml -> paths.partnet_root to your PartNet data directory
```

## Project Structure

```
3DPART/
├── configs/                      # YAML configuration
├── notebooks/                    # Run in order 00 → 10
│   ├── 00_env_check.ipynb
│   ├── 01_build_dataset.ipynb
│   ├── 02_extract_boundary_loops.ipynb
│   ├── 03_target_scoring_baseline.ipynb
│   ├── 04_repair_backends.ipynb
│   ├── 05_evaluation.ipynb            # Tables 2–5
│   ├── 06_ablation_target_selection.ipynb
│   ├── 07_multicategory_experiments.ipynb
│   ├── 08_make_tables_and_figures.ipynb  # All paper figures
│   ├── 09_sota_comparison.ipynb         # SOTA + distance metrics
│   └── 10_learning_selection.ipynb      # Learning-based selection
├── src/
│   ├── baselines/               # SOTA methods
│   │   ├── trimesh_fill.py     #   trimesh built-in hole filling
│   │   ├── open3d_fill.py      #   Poisson + ball-pivoting recon
│   │   ├── advancing_front.py  #   Liepa-style advancing front
│   │   └── largest_hole.py     #   Naive largest-hole baseline
│   ├── data/                    # Dataset construction & loading
│   ├── evaluation/              # Metrics
│   │   ├── metrics_closure.py
│   │   ├── metrics_complexity.py
│   │   ├── metrics_quality.py
│   │   ├── metrics_locality.py
│   │   ├── metrics_distance.py #   Chamfer / Hausdorff / deviation
│   │   └── evaluator.py        #   Unified 5-dimensional evaluator
│   ├── experiments/
│   ├── geometry/                # Boundary, triangulation, quality
│   ├── io/
│   ├── repair/                  # Center-fan, planar, minimal-area
│   ├── target_selection/        # Geometric + learned selection
│   │   ├── selectors.py        #   BBox proximity, largest-hole
│   │   ├── learning.py         #   RF, MLP, GBDT classifiers
│   │   └── ...
│   └── visualization/           # Mesh rendering + plot utilities
│       ├── mesh_renderer.py    #   Open3D + matplotlib rendering
│       └── plot_utils.py       #   Paper-style plots
└── outputs/
    ├── figures/                 # PDF figures
    ├── tables/                  # CSV + LaTeX tables
    └── metrics/                 # Raw evaluation data
```

## Quick Start

Run notebooks 00 → 10 sequentially:

| Notebook | Purpose | Paper Content |
|----------|---------|---------------|
| 00 | Environment check | — |
| 01 | Build dataset from PartNet | §3.2 |
| 02 | Boundary loop analysis | §3.3 |
| 03 | Target selection scoring | §3.4 |
| 04 | Test repair backends | §3.5 |
| **05** | **Full evaluation (ours)** | **Tables 2–5** |
| 06 | Ablation: target selection | §4.4 |
| 07 | Multi-category extension | §4+ |
| **08** | **All paper figures + tables** | **Figs 2–5, LaTeX** |
| **09** | **SOTA comparison + distance** | **Table 6–7, Fig SOTA** |
| **10** | **Learning-based selection** | **RF/MLP/GBDT experiment** |

## Methods Compared

| Method | Target Selection | Triangulation | New Vertices | Type |
|--------|-----------------|---------------|--------------|------|
| Center-fan (ours) | Removed-part-aware | Fan from centroid | Yes | Ours |
| Planar + RPA (ours) | Removed-part-aware | Delaunay + filtering | No | Ours |
| Planar + LH-only | Largest hole | Delaunay + filtering | No | Ablation |
| Trimesh fill-all | All holes | trimesh built-in | Varies | SOTA |
| Advancing-front + RPA | Removed-part-aware | Advancing front | Optional | SOTA |
| Poisson recon | N/A (global) | Poisson surface | Yes | SOTA |

## Evaluation Metrics

| Dimension | Metrics |
|-----------|---------|
| Closure | Residual target loop length, improvement |
| Complexity | New vertices, new faces |
| Quality | Mean/min triangle quality |
| Locality | Faces inside/outside repair zone, locality ratio |
| Distance | Chamfer distance, Hausdorff distance, surface deviation |

## Citation

```bibtex
@article{ji2025repair,
  title={Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change},
  author={Ji, Bingyang and Gao, Changxin and Chen, Fuhao and Cui, Shan},
  journal={The Visual Computer},
  year={2025}
}
```

## License

This code is provided for research purposes.
