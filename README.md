# 3D-Part-Repair

Code, experiment pipeline, tables, trained models, and figure-generation utilities for the manuscript:

> **Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change**submitted to *The Visual Computer*

This repository is intended to make the manuscript reproducible. It includes:

* PartNet-based dataset construction
* target-loop selection (geometric and learned)
* local mesh repair baselines
* evaluation metrics and table generation
* multi-category supplementary experiments
* paper figure rendering utilities

If you use this repository, please cite the manuscript and, when available, the archived Zenodo release.

* * *

## 1. What this repository corresponds to

The **paper-facing scope** of the repository is:

* **Primary benchmark**: Chair / leg removal
* **Supplementary cross-category settings**: Table / leg removal and StorageFurniture / door removal
* **Repair methods used in the manuscript**:
  * Center-fan + removed-part-aware (RPA)
  * Planar triangulation + RPA
  * Planar triangulation + largest-hole-only (LH-only)
  * Advancing-front + RPA
  * Trimesh fill-all
  * Open3D Poisson reconstruction
* **Learning-based extension used in the manuscript**:
  * target-loop classification with **Random Forest / MLP / GBDT**

The repository also contains some **exploratory / auxiliary items** that are not central to the current paper narrative, such as:

* `src/repair/mlp_patch.py`
* notebook `11_mlp_patch_training.ipynb`
* extra configs such as `bed_leg.yaml` and `storagefurniture_foot.yaml`

These can be kept in the release, but they should be described as **supplementary exploratory code**, not as the paper's main contribution.

* * *

## 2. Repository structure

    3D-Part-Repair/
    ├── configs/
    │   ├── default.yaml
    │   ├── chair_leg.yaml
    │   ├── table_leg.yaml
    │   ├── multi_category.yaml
    │   ├── storagefurniture_foot.yaml
    │   └── bed_leg.yaml
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
    │   ├── 02_extract_boundary_loops.ipynb
    │   ├── 03_target_scoring_baseline.ipynb
    │   ├── 04_repair_backends.ipynb
    │   ├── 05_evaluation.ipynb
    │   ├── 06_ablation_target_selection.ipynb
    │   ├── 07_multicategory_experiments.ipynb
    │   ├── 08_make_tables_and_figures.ipynb
    │   ├── 09_sota_comparison.ipynb
    │   ├── 10_learning_selection.ipynb
    │   └── 11_mlp_patch_training.ipynb
    ├── outputs/
    │   ├── Chair_leg/
    │   ├── Table_leg/
    │   ├── StorageFurniture_door/
    │   ├── figures/
    │   ├── logs/
    │   ├── metrics/
    │   ├── models/
    │   ├── repaired_meshes/
    │   └── tables/
    ├── scripts/
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
    │   ├── config.py
    │   └── utils.py
    ├── render_paper_figures.py
    ├── requirements.txt
    └── README.md

* * *

## 3. Environment and installation

### 3.1 Requirements

Python dependencies are listed in `requirements.txt`:

* numpy
* scipy
* trimesh
* pyyaml
* tqdm
* pandas
* matplotlib
* scikit-learn
* open3d
* Pillow

### 3.2 Installation

    git clone https://github.com/jibingyang11/3D-Part-Repair.git
    cd 3D-Part-Repair
    pip install -r requirements.txt

* * *

## 4. Data requirement: PartNet

This project assumes access to **PartNet data v0**.

Edit `configs/default.yaml` and set:

    paths:
      partnet_root: "E:/datasets/partnet"

The notebooks build paired samples from PartNet into the directory specified by:

    paths:
      raw_data_dir: ...

In this project, the generated benchmark data are loaded from `raw_data_dir` by the loaders and rendering scripts.

* * *

## 5. Configuration files

### 5.1 Main configs used by the manuscript

#### `configs/chair_leg.yaml`

Primary benchmark for the paper.

* category: `Chair`
* removed part: `leg`
* sample count: `100`

#### `configs/table_leg.yaml`

Supplementary cross-category experiment.

* category: `Table`
* removed part: `leg`
* sample count: `50`

#### `configs/multi_category.yaml`

Paper-facing multi-category configuration used for supplementary experiments.

* Chair / leg / 100
* Table / leg / 50
* StorageFurniture / door / 50

### 5.2 Exploratory configs included in the repo

#### `configs/storagefurniture_foot.yaml`

Exploratory alternative config for StorageFurniture.

#### `configs/bed_leg.yaml`

Exploratory alternative config for Bed / leg removal.

These are useful for extending the benchmark, but the current LaTeX manuscript should not claim quantitative results from them unless corresponding outputs are actually reported.

* * *

## 6. Recommended notebook execution order

### 6.1 Environment check

1. `00_env_check.ipynb`

### 6.2 Dataset construction

Use the notebook that matches the category you want to build:

* `01_build_dataset_chair.ipynb`
* `01_build_dataset_table.ipynb`
* `01_build_dataset_storagefurniture.ipynb`
* `01_build_dataset_bed.ipynb.ipynb` (exploratory; consider renaming to `01_build_dataset_bed.ipynb` before release)

### 6.3 Geometry and repair workflow

2. `02_extract_boundary_loops.ipynb`
3. `03_target_scoring_baseline.ipynb`
4. `04_repair_backends.ipynb`
5. `05_evaluation.ipynb`
6. `06_ablation_target_selection.ipynb`
7. `07_multicategory_experiments.ipynb`
8. `08_make_tables_and_figures.ipynb`
9. `09_sota_comparison.ipynb`
10. `10_learning_selection.ipynb`

### 6.4 Exploratory learning-based patch generator

11. `11_mlp_patch_training.ipynb`

This notebook is best described as **supplementary exploratory code**, not as the core learning contribution of the current manuscript.

* * *

## 7. Core method components

### 7.1 Dataset construction

Implemented in:

* `src/data/dataset_builder.py`
* `src/data/dataset_index.py`
* `src/data/sample_loader.py`

The benchmark stores, for each sample:

* complete mesh
* damaged mesh
* removed-part mesh
* metadata

### 7.2 Target-loop selection

Implemented in:

* `src/target_selection/selectors.py`
* `src/target_selection/features.py`
* `src/target_selection/labeling.py`
* `src/target_selection/learning.py`

Paper-facing target-selection settings:

* geometric removed-part-aware target loop selection
* largest-hole-only ablation
* learned target-loop classification (RF / MLP / GBDT)

### 7.3 Repair methods

Implemented in:

* `src/repair/center_fan.py`
* `src/repair/planar_patch.py`
* `src/baselines/advancing_front.py`
* `src/baselines/trimesh_fill.py`
* `src/baselines/open3d_fill.py`

Supplementary exploratory implementation:

* `src/repair/mlp_patch.py`

### 7.4 Evaluation

Implemented in:

* `src/evaluation/evaluator.py`
* `src/evaluation/metrics_closure.py`
* `src/evaluation/metrics_complexity.py`
* `src/evaluation/metrics_quality.py`
* `src/evaluation/metrics_locality.py`
* `src/evaluation/metrics_distance.py`

Paper-facing evaluation dimensions:

1. closure
2. patch complexity
3. triangle quality
4. locality
5. geometric distance (Chamfer / Hausdorff)

* * *

## 8. Running experiments from code

### 8.1 Single-sample experiment

Use `src/experiments/run_single.py` through notebooks or import it in Python:

    from src.experiments.run_single import run_single_experiment
    
    result = run_single_experiment(
        sample_id="35123",
        data_dir="data/raw/chair_leg_removal",
        output_dir="outputs/repaired_meshes",
        margin=0.05,
        proximity_threshold=0.1,
        save_meshes=True,
        include_distance=True,
        include_sota=True,
    )

### 8.2 Batch experiment

    from src.experiments.run_batch import run_batch_experiment
    
    df = run_batch_experiment(
        data_dir="data/raw/chair_leg_removal",
        output_dir="outputs",
        margin=0.05,
        proximity_threshold=0.1,
        save_meshes=True,
        include_distance=True,
        include_sota=True,
    )

### 8.3 Table generation

    from src.experiments.summarize import summarize_results
    import pandas as pd
    
    df = pd.read_csv("outputs/metrics/all_results.csv")
    tables = summarize_results(df, "outputs")

* * *

## 9. Mapping repository outputs to paper tables

The repository already contains generated CSV tables in `outputs/tables/`.

### Main output tables

* `table2_quantitative.csv`Ours-only quantitative comparison
  
* `table3_locality.csv`Locality analysis
  
* `table4_failures.csv`Failure cases of LH-only selection
  
* `table5_target_selection.csv`Target-selection comparison
  
* `table6_sota_comparison.csv`Full comparison including SOTA baselines
  
* `table7_distance.csv`Distance metrics (Chamfer / Hausdorff / deviation)
  
* `table8_mlp_comparison.csv`Supplementary MLP patch comparison
  
* `table_multi_category.csv`Cross-category summary
  

### Important note

The current manuscript uses the **full SOTA comparison table** and the **distance table** as its primary quantitative reporting source. If you update the manuscript tables manually, prefer values from:

* `outputs/tables/table6_sota_comparison.csv`
* `outputs/tables/table7_distance.csv`
* `outputs/tables/table_multi_category.csv`

For `table5_target_selection.csv`, the repository exports raw `Outside ↓` counts. If the manuscript reports **Outside Ratio** instead, note in the paper that this is a normalized value derived from the raw counts and total new faces.

* * *

## 10. Figure rendering

Use `render_paper_figures.py` to generate paper-quality images.

Typical usage:

    python render_paper_figures.py --config configs/chair_leg.yaml --sample_id 35123 --component_index 0 --rebuild_single_part --name_prefix chair --reset_camera

The script supports:

* interactive first-view camera selection
* camera reuse for later renders
* target-loop visualization
* repaired patch visualization
* failure-case rendering
* comparison strips

### Release note before uploading

Before publishing the repository, check `render_paper_figures.py` carefully:

1. `FAILURE_CASES` still contains placeholder sample IDs for non-chair categories.
2. In the standard non-`--render_failure_triplet` path, make sure `cfg = load_config(...)` is executed before `ensure_dirs(cfg)` and `DatasetIndex(...)` are used.
3. If you want to keep StorageFurniture / door failure rendering, provide a matching config file or change the script to point to an existing config.

These are small cleanup items, but they should be fixed before the public release.

* * *

## 11. Reproducibility and release checklist

Before pushing to GitHub / Zenodo, verify the following:

* [ ] `README.md` is updated and matches the actual repository structure
* [ ] manuscript-facing configs are present and correct
* [ ] notebook names are consistent
* [ ] `render_paper_figures.py` runs without placeholder IDs or config mismatches
* [ ] `requirements.txt` is complete
* [ ] `outputs/tables/` contains the exact CSVs used to fill the paper
* [ ] the manuscript abstract contains the GitHub link
* [ ] the repository states that it is directly related to the manuscript submitted to *The Visual Computer*
* [ ] the Zenodo DOI points to the exact archived release used in the paper

* * *

## 12. Alignment with the current manuscript

The current LaTeX manuscript is aligned with the repository **if you describe the repository in the following way**:

* the paper's main learning contribution is **learning-based target-loop classification**, not MLP patch generation
* the paper's main reported cross-category setting is **Chair / Table / StorageFurniture-door**
* MLP patch code exists in the repo, but should be labeled as **supplementary exploratory code** unless the manuscript is expanded to report it centrally

* * *

## 13. Citation

    @article{ji2026targetaware,
      title   = {Target-Aware Local Mesh Repair for Semantic Part Removal with Minimal Geometric Change},
      author  = {Ji, Bingyang and Gao, Changxin and Chen, Fuhao and Cui, Shan},
      journal = {The Visual Computer},
      year    = {2026},
      note    = {submitted}
    }

If you archive a Zenodo release, also add the DOI-based software citation here.

* * *

## 14. License

Add the license you intend to release under before making the repository public.

For example:

* MIT License
* BSD-3-Clause
* CC BY-NC 4.0 for benchmark metadata / figures, if you want a different data-license policy

If you do not yet have a final license decision, do **not** leave the repository ambiguous before public release.
