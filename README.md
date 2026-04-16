# Experiments README

This document explains how to reproduce the experiments for the manuscript:

**Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change**

This repository is directly associated with the manuscript currently submitted to **The Visual Computer**. If you use this code or the released benchmark settings, please cite the associated manuscript and the archived release.

## 1. What this repository contains

The experimental codebase includes:

* **Dataset construction** from raw PartNet mesh data
* **Target-aware local repair** after semantic part removal
* **Geometric repair methods**:
  * `center_fan`
  * `planar_removed_part_aware`
  * `planar_largest_hole_only`
* **Representative comparison baselines**:
  * `advancing_front_rpa`
  * `trimesh_fill_all`
  * `open3d_poisson`
* **Learning-based target-loop selection**:
  * Random Forest
  * MLP
  * GBDT
* **Five-dimensional evaluation**:
  * closure
  * complexity
  * quality
  * locality
  * geometric distance
* **Cross-category experiments**:
  * Chair / leg
  * Table / leg
  * StorageFurniture / door
* **Preliminary multi-part pilot**:
  * Chair / two-leg removal
  * Table / two-leg removal

## 2. Repository structure

    3D-Part-Repair/
    ├── configs/
    │   ├── default.yaml
    │   ├── chair_leg.yaml
    │   ├── chair_two_leg.yaml
    │   ├── table_leg.yaml
    │   ├── table_two_leg.yaml
    │   ├── storagefurniture_foot.yaml
    │   ├── bed_leg.yaml
    │   └── multi_category.yaml
    ├── data/
    │   ├── raw/
    │   ├── processed/
    │   └── splits/
    ├── notebooks/
    │   ├── 00_env_check.ipynb
    │   ├── 01_build_dataset_chair.ipynb
    │   ├── 01_build_dataset_table.ipynb
    │   ├── 01_build_dataset_storagefurniture.ipynb
    │   ├── 01_build_dataset_bed.ipynb.ipynb
    │   ├── 01_build_dataset_chair_two_leg.ipynb
    │   ├── 01_build_dataset_table_two_leg.ipynb
    │   ├── 02_extract_boundary_loops.ipynb
    │   ├── 03_target_scoring_baseline.ipynb
    │   ├── 04_repair_backends.ipynb
    │   ├── 05_evaluation.ipynb
    │   ├── 06_ablation_target_selection.ipynb
    │   ├── 07_multicategory_experiments.ipynb
    │   ├── 07_multipart_pilot.ipynb
    │   ├── 08_make_tables_and_figures.ipynb
    │   ├── 09_sota_comparison.ipynb
    │   ├── 10_learning_selection.ipynb
    │   └── 11_mlp_patch_training.ipynb
    ├── outputs/
    │   ├── Chair_leg/
    │   ├── Chair_leg_mlp/
    │   ├── Table_leg/
    │   ├── StorageFurniture_door/
    │   ├── chair_two_leg_removal/
    │   ├── table_two_leg_removal/
    │   ├── figures/
    │   ├── metrics/
    │   ├── tables/
    │   └── models/
    ├── src/
    │   ├── baselines/
    │   ├── data/
    │   ├── evaluation/
    │   ├── experiments/
    │   ├── geometry/
    │   ├── io/
    │   ├── repair/
    │   ├── target_selection/
    │   ├── visualization/
    │   └── config.py
    ├── render_paper_figures.py
    ├── requirements.txt
    └── README.md

## 3. Environment setup

### 3.1 Python

Recommended:

* Python 3.9–3.11
* Windows or Linux
* Conda environment recommended

### 3.2 Install dependencies

    pip install -r requirements.txt

Main dependencies:

* `numpy`
* `scipy`
* `trimesh`
* `pandas`
* `matplotlib`
* `scikit-learn`
* `open3d`
* `Pillow`
* `PyYAML`
* `tqdm`

## 4. Data preparation

### 4.1 Raw source data

This project uses **PartNet data v0** as the raw source dataset.

Set the PartNet root in:

    configs/default.yaml

Example:

    paths:
      partnet_root: "E:/datasets/partnet"

### 4.2 Important note on data redistribution

This repository does **not** redistribute the original PartNet data. Users must obtain the original dataset separately and then run the provided dataset-construction notebooks/scripts.

## 5. Configuration files

All experiments are driven by YAML configuration files in `configs/`.

### Main single-part settings

* `chair_leg.yaml`
* `table_leg.yaml`
* `storagefurniture_foot.yaml` / `storagefurniture_door`-related outputs
* `bed_leg.yaml` (exploratory / optional)

### Preliminary multi-part pilot settings

* `chair_two_leg.yaml`
* `table_two_leg.yaml`

Key fields include:

* `dataset.category`
* `dataset.semantic_label`
* `dataset.total_samples`
* `dataset.num_parts_to_remove`
* `repair.margin`
* `repair.proximity_threshold`
* `repair.methods`

## 6. Recommended experiment order

### Step 0. Environment check

Run:

* `notebooks/00_env_check.ipynb`

### Step 1. Build datasets

Single-part:

* `01_build_dataset_chair.ipynb`
* `01_build_dataset_table.ipynb`
* `01_build_dataset_storagefurniture.ipynb`

Preliminary multi-part pilot:

* `01_build_dataset_chair_two_leg.ipynb`
* `01_build_dataset_table_two_leg.ipynb`

These notebooks construct paired samples containing:

* `complete.obj`
* `damaged.obj`
* `removed_part.obj`
* `meta.json`

### Step 2. Boundary and target inspection

* `02_extract_boundary_loops.ipynb`
* `03_target_scoring_baseline.ipynb`

### Step 3. Repair backends and basic evaluation

* `04_repair_backends.ipynb`
* `05_evaluation.ipynb`
* `06_ablation_target_selection.ipynb`

### Step 4. Additional experiments

* `07_multicategory_experiments.ipynb`
* `07_multipart_pilot.ipynb`
* `09_sota_comparison.ipynb`
* `10_learning_selection.ipynb`

### Step 5. Paper tables and figures

* `08_make_tables_and_figures.ipynb`
* `render_paper_figures.py`

## 7. Main methods used in the paper

### 7.1 Target-aware local repair

* `center_fan`
* `planar_removed_part_aware`

### 7.2 Ablation

* `planar_largest_hole_only`

### 7.3 Representative comparison baselines

* `advancing_front_rpa`
* `trimesh_fill_all`
* `open3d_poisson`

### 7.4 Learning-based target selection

Implemented in the learning notebooks / target-selection modules:

* Random Forest
* MLP
* GBDT

## 8. How results are stored

Batch experiments typically save outputs under:

    outputs/<experiment_name>/
    ├── logs/
    ├── metrics/
    │   ├── all_results.csv
    │   ├── all_results.json
    │   └── run_summary.json
    └── repaired_meshes/   (optional)

The flattened metrics file:

    outputs/<experiment_name>/metrics/all_results.csv

contains per-sample results with columns such as:

* `planar_removed_part_aware/closure_residual`
* `planar_removed_part_aware/improvement`
* `planar_removed_part_aware/locality_ratio`
* `advancing_front_rpa/chamfer_distance`
* etc.

## 9. Mapping repository outputs to paper experiments

### Main single-part benchmark

Primary output folders:

* `outputs/Chair_leg/`
* `outputs/metrics/`

### Learning-based target selection

* `outputs/Chair_leg_mlp/`
* `outputs/models/loop_clf_MLP.pkl`

### Cross-category extension

* `outputs/Table_leg/`
* `outputs/StorageFurniture_door/`

### Preliminary multi-part pilot

* `outputs/chair_two_leg_removal/`
* `outputs/table_two_leg_removal/`

### Final tables

* `outputs/tables/`

## 10. Reproducing the preliminary multi-part pilot

### Chair two-leg pilot

1. Build dataset with `01_build_dataset_chair_two_leg.ipynb`
2. Run `07_multipart_pilot.ipynb` with `CONFIG_NAME = "chair_two_leg.yaml"`
3. Read results from:

    outputs/chair_two_leg_removal/metrics/all_results.csv
    outputs/tables/chair_two_leg_removal_multipart_pilot_summary.csv

### Table two-leg pilot

1. Build dataset with `01_build_dataset_table_two_leg.ipynb`
2. Run `07_multipart_pilot.ipynb` with `CONFIG_NAME = "table_two_leg.yaml"`
3. Read results from:

    outputs/table_two_leg_removal/metrics/all_results.csv
    outputs/tables/table_two_leg_removal_multipart_pilot_summary.csv

## 11. Figure generation

Qualitative figures are rendered with:

    python render_paper_figures.py

This script is used to generate visualizations for:

* target loops
* repaired patches
* failure-case comparisons
* paper figure panels

## 12. Important clarification about learning components

The manuscript’s **main learning contribution** is the **learning-based target-loop selection** (RF / MLP / GBDT).

The repository also contains exploratory code for **MLP-based patch generation** (`src/repair/mlp_patch.py`, `11_mlp_patch_training.ipynb`). This component is included for completeness and experimentation, but the paper’s central claim is **not** that a learned patch generator is the main contribution. The core message of the manuscript is that **correct target selection is more important than the specific local triangulation backend**.

## 13. Reproducibility notes

To reproduce the reported results as closely as possible:

1. Use the same PartNet source version
2. Keep the configuration files unchanged
3. Use the provided train/val/test split files in `data/splits/`
4. Use the metrics from the saved `all_results.csv` files
5. Generate final paper tables from `outputs/tables/`

## 14. Archived release and citation

GitHub repository:

* `https://github.com/jibingyang11/3D-Part-Repair`

Archived release / DOI:

* `https://doi.org/10.5281/zenodo.19533166`

This archived release is the exact version associated with the manuscript submitted to **The Visual Computer**.

## 15. Suggested citation note

If you use this repository, please cite:

1. The associated manuscript
2. The archived Zenodo release

## 16. Known practical notes

* `open3d_poisson` is typically the slowest baseline
* distance metrics significantly increase runtime
* multi-part pilot experiments are intentionally smaller in scale than the primary single-part benchmark
* some exploratory configs/files are included in the repository but are not all equally central to the final manuscript narrative

## 17. Contact

For questions about the code, benchmark construction, or reproduction of the paper results, please contact the corresponding author listed in the manuscript.
