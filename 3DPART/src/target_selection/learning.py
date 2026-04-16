"""Learning-based target boundary loop selection.

Provides three lightweight classifiers to learn which boundary loop
corresponds to the removed semantic part:

1. RandomForest + hand-crafted features (interpretable)
2. MLP on the same features (more expressive)
3. GBDT (gradient boosted decision trees)

Both are trained on labeled (loop, is_target) pairs produced by the
geometric labeling module and can replace or augment the bbox-proximity
heuristic at inference time.

Design connections to related work:
  - The dual-network / ensemble approach draws on insights from robust
    learning under noisy conditions (cf. Zhang et al., Joint-Learning,
    CAVW 2025, which shows dual-network frameworks improve robustness
    for 3D point cloud segmentation under label noise).
  - Feature engineering for boundary loops leverages both local geometric
    and global shape properties, paralleling multi-scale feature designs
    in 3D understanding (cf. Xue & Wang, LGNet, CAVW 2025).
  - The boundary-level feature extraction connects to part-aware geometric
    reasoning used in hybrid 3D model analysis (cf. Uwimana et al.,
    VR&IH 2025 on hybrid representation for CAD segmentation).
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Tuple, Optional

import trimesh

from .features import extract_loop_features
from .labeling import label_loops

try:
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import cross_val_score
    from sklearn.metrics import classification_report, accuracy_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False


# ------------------------------------------------------------------ #
#  Feature / label collection helpers
# ------------------------------------------------------------------ #

def collect_training_data(
    sample_ids: List[str],
    data_dir: str,
    margin: float = 0.05,
    threshold: float = 0.1,
) -> Tuple[List[Dict[str, float]], List[int]]:
    """Collect feature vectors and labels from a list of samples.

    Returns (features_list, labels) where each element corresponds
    to one boundary loop.
    """
    from ..data.sample_loader import SampleLoader
    from ..geometry.boundary import extract_boundary_loops

    loader = SampleLoader(data_dir)
    all_features = []
    all_labels = []

    for sid in sample_ids:
        try:
            sample = loader.load(sid)
            damaged = sample["damaged_mesh"]
            removed = sample["removed_part_mesh"]

            loops = extract_boundary_loops(damaged)
            if not loops:
                continue

            labels_info = label_loops(damaged, loops, removed, margin, threshold)

            for i, loop in enumerate(loops):
                feats = extract_loop_features(damaged, loop, removed)
                all_features.append(feats)
                all_labels.append(labels_info[i]["is_target"])

        except Exception:
            continue

    return all_features, all_labels


def _dicts_to_matrix(features_list: List[Dict[str, float]]) -> Tuple[np.ndarray, List[str]]:
    """Convert feature dicts to a numpy matrix and return column names."""
    if not features_list:
        return np.empty((0, 0)), []
    keys = sorted(features_list[0].keys())
    X = np.array([[d[k] for k in keys] for d in features_list], dtype=np.float64)
    return X, keys


# ------------------------------------------------------------------ #
#  Classifiers
# ------------------------------------------------------------------ #

class LoopClassifierRF:
    """Random Forest classifier for target vs non-target loops."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required")
        self.scaler = StandardScaler()
        self.model = RandomForestClassifier(
            n_estimators=n_estimators, random_state=random_state,
            class_weight="balanced", max_depth=10,
        )
        self.feature_names = []
        self.fitted = False

    def fit(self, features_list: List[Dict[str, float]], labels: List[int]):
        X, self.feature_names = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, features_list: List[Dict[str, float]]) -> List[int]:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()

    def predict_proba(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def feature_importance(self) -> Dict[str, float]:
        if not self.fitted:
            return {}
        imp = self.model.feature_importances_
        return {n: float(v) for n, v in zip(self.feature_names, imp)}

    def cross_validate(self, features_list, labels, cv=5) -> Dict[str, float]:
        X, _ = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="f1")
        return {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model,
                         "feature_names": self.feature_names}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.fitted = True


class LoopClassifierMLP:
    """MLP classifier for target vs non-target loops."""

    def __init__(self, hidden_sizes: Tuple = (64, 32), random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required")
        self.scaler = StandardScaler()
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_sizes, random_state=random_state,
            max_iter=500, early_stopping=True, validation_fraction=0.15,
            learning_rate="adaptive",
        )
        self.feature_names = []
        self.fitted = False

    def fit(self, features_list: List[Dict[str, float]], labels: List[int]):
        X, self.feature_names = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, features_list: List[Dict[str, float]]) -> List[int]:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()

    def predict_proba(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def cross_validate(self, features_list, labels, cv=5) -> Dict[str, float]:
        X, _ = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="f1")
        return {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler, "model": self.model,
                         "feature_names": self.feature_names}, f)

    def load(self, path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scaler = data["scaler"]
        self.model = data["model"]
        self.feature_names = data["feature_names"]
        self.fitted = True


class LoopClassifierGBDT:
    """Gradient Boosted Decision Trees for target loop selection."""

    def __init__(self, n_estimators: int = 100, random_state: int = 42):
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required")
        self.scaler = StandardScaler()
        self.model = GradientBoostingClassifier(
            n_estimators=n_estimators, random_state=random_state,
            max_depth=5, learning_rate=0.1,
        )
        self.feature_names = []
        self.fitted = False

    def fit(self, features_list: List[Dict[str, float]], labels: List[int]):
        X, self.feature_names = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        self.fitted = True

    def predict(self, features_list: List[Dict[str, float]]) -> List[int]:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled).tolist()

    def predict_proba(self, features_list: List[Dict[str, float]]) -> np.ndarray:
        X, _ = _dicts_to_matrix(features_list)
        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)[:, 1]

    def cross_validate(self, features_list, labels, cv=5) -> Dict[str, float]:
        X, _ = _dicts_to_matrix(features_list)
        y = np.array(labels)
        X_scaled = self.scaler.fit_transform(X)
        scores = cross_val_score(self.model, X_scaled, y, cv=cv, scoring="f1")
        return {"f1_mean": float(scores.mean()), "f1_std": float(scores.std())}


# ------------------------------------------------------------------ #
#  Inference helper: use a trained classifier to select target loops
# ------------------------------------------------------------------ #

def select_loops_by_classifier(
    mesh: trimesh.Trimesh,
    loops: List[List[int]],
    removed_part_mesh: trimesh.Trimesh,
    classifier,
    prob_threshold: float = 0.5,
) -> List[List[int]]:
    """Use a trained classifier to select target boundary loops.

    Falls back to the highest-probability loop if none exceed the
    threshold.
    """
    if not loops:
        return []

    features_list = [extract_loop_features(mesh, loop, removed_part_mesh)
                     for loop in loops]

    probs = classifier.predict_proba(features_list)

    selected = [loops[i] for i, p in enumerate(probs) if p >= prob_threshold]

    if not selected:
        best_idx = int(np.argmax(probs))
        selected = [loops[best_idx]]

    return selected
